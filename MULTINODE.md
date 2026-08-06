# Qwen19B slime 多机交付手册

这套多机入口不依赖 SSH、SLURM 或 Kubernetes。每台服务器运行同一 Docker 镜像和一份 Ray 节点进程；只在 head 节点提交一次训练。原来的单机命令保持可用。

## 1. 交付前提

当前脚本采用同构节点：每台服务器必须向 Ray 暴露相同数量的 GPU。还需满足：

- 节点间有稳定的私网 IP，容器能互相访问；建议 host network。
- 每台机器使用完全相同的镜像 digest、slime Git revision 和项目代码。
- 项目、模型、训练数据、验证数据在每台机器上的绝对路径一致。模型和只读数据可以是共享存储，也可以逐机复制，但文件内容必须一致。
- `OUTPUT_ROOT` 必须是所有节点都能写入的共享文件系统，checkpoint 不能只存在于 head 的本地盘。
- 节点间防火墙允许 Ray、NCCL/Gloo 和 GPU 通信流量。最省事的做法是只在可信私网内允许节点彼此全端口通信；不要把 Ray dashboard 或 GCS 暴露到公网。
- 主机时间同步，GPU driver 与硬件代际兼容。正式训练前先用 `nvidia-smi` 和 NCCL 测试确认跨机链路。
- 每台服务器的 hostname 应唯一；管理脚本默认沿用宿主机短 hostname，也可用 `SLIME_CONTAINER_HOSTNAME` 显式指定。

交付目录建议在所有机器上保持 `/home/user01/jx/19b-rl-slime`。如果接收方使用其他路径，必须所有节点一起修改，不能只改 head。

## 2. 每台服务器准备容器

在每台服务器执行：

```bash
cd /home/user01/jx/19b-rl-slime
export SLIME_CONTAINER_NAME=slime-qwen19b-multinode
export SLIME_DOCKER_NETWORK=host
bash scripts/docker_env.sh pull
bash scripts/docker_env.sh create
bash scripts/docker_env.sh status
```

`create` 不会覆盖已有容器。旧的 `slime-qwen19b-dev` 如果是 bridge network，可以继续用于单机；多机请使用上面新的容器名，避免删除或重建旧容器。

宿主机项目目录以 bind mount 方式挂载，因此可以直接修改代码；修改会实时出现在容器里。若切换了 `third_party/slime` 的分支或重建源码目录，再执行：

```bash
SLIME_CONTAINER_NAME=slime-qwen19b-multinode \
bash scripts/docker_env.sh install-source
```

## 3. 配置两台 8-GPU 节点

先在每台机器的项目目录复制模板并按实际情况修改：

```bash
cp configs/multinode.env.example configs/multinode.env
```

模板默认：

- head 私网 IP：`10.0.0.10`
- 2 台服务器，每台 8 GPU
- actor 总 GPU 数：16
- Megatron：TP=2、PP=2、CP=1、EP=2
- SGLang rollout TP=2

`SOCKET_IFNAME` 必须改为承载跨机流量的真实网卡，如 `ib0`、`bond0` 或 `eth0`。可用 `ip -br addr` 查看。所有节点的公共配置相同，只有 `NODE_IP` 各不相同。

> `TP=2, PP=2, EP=2` 是 2×8 H100 的候选起点。增加机器并不会自动降低每个 rank 的显存；正式长度仍需依据显存和链路调整 PP/TP、rollout 长度和 SGLang 静态显存比例。

## 4. 启动 Ray 集群

以下命令都在宿主机执行，管理脚本会把环境变量传入容器。

先在 head 节点启动：

```bash
cd /home/user01/jx/19b-rl-slime
set -a
source configs/multinode.env
set +a
export NODE_IP="$MASTER_ADDR"
export SLIME_CONTAINER_NAME=slime-qwen19b-multinode
bash scripts/docker_env.sh ray-head
```

然后在每个 worker 节点启动，把 `NODE_IP` 替换成本机私网 IP：

```bash
cd /home/user01/jx/19b-rl-slime
set -a
source configs/multinode.env
set +a
export NODE_IP=10.0.0.11
export SLIME_CONTAINER_NAME=slime-qwen19b-multinode
bash scripts/docker_env.sh ray-worker
```

脚本默认不会停止或覆盖已经运行的 Ray。只有明确要重启本机 Ray 时才增加 `RESTART_RAY=1`。集群状态可在 head 查看：

```bash
set -a
source configs/multinode.env
set +a
SLIME_CONTAINER_NAME=slime-qwen19b-multinode \
bash scripts/docker_env.sh ray-status
```

状态里应看到 2 个 alive nodes 和总计 16 GPU。

## 5. W&B 登录

如果使用 W&B online 模式，在每台服务器的容器中各登录一次：

```bash
SLIME_CONTAINER_NAME=slime-qwen19b-multinode \
bash scripts/docker_env.sh shell
wandb login
```

Ray worker 可能在任意节点执行训练 actor，因此只登录 head 不够。本项目不会把 `WANDB_API_KEY` 写入 Ray runtime env 或日志。也可以由接收方的 secret manager 在每个容器中注入凭据。

## 6. 提交 multi-node smoke

只在 head 容器里执行一次：

```bash
SLIME_CONTAINER_NAME=slime-qwen19b-multinode \
bash scripts/docker_env.sh shell

set -a
source configs/multinode.env
set +a
export NODE_IP="$MASTER_ADDR"

MODE=smoke \
EXPERIMENT_NAME=qwen19b_multinode_smoke_$(date +%Y%m%d_%H%M%S) \
bash run_dapo.sh \
  --use-wandb \
  --wandb-mode online \
  --wandb-team "iceblwdzs" \
  --wandb-project "qwen19b-slime-dapo" \
  --wandb-group "qwen19b-multinode-smoke" \
  --wandb-dir "/mnt/data/user01/19b100w/slime_runs/wandb"
```

多机模式下 `run_dapo.sh` 会在提交作业前自动检查：

- Ray alive node 数和每节点 GPU 资源数；
- 每台节点上的模型配置/可选索引、数据、项目 hook 和 slime 文件是否存在且签名一致；
- 模型索引引用的每个权重 shard（无索引时为目录内 safetensors）是否存在，且各节点 shard 名称与尺寸一致；
- slime Git revision 是否一致；
- 每台节点能否看到足够物理 GPU；
- `OUTPUT_ROOT` 是否可写，并通过临时文件确认它确实是所有节点看到的同一共享文件系统。

检查失败时不会开始加载 19B 模型。若只是想打印训练命令而不连 Ray，可加 `DRY_RUN=1`。

## 7. 正式训练

multi-node smoke 完整跑完、保存 checkpoint 且 W&B 指标正常后，仍只在 head 容器提交：

```bash
set -a
source configs/multinode.env
set +a
export NODE_IP="$MASTER_ADDR"

MODE=train \
EXPERIMENT_NAME=qwen19b_multinode_dapo \
bash run_dapo.sh \
  --use-wandb \
  --wandb-mode online \
  --wandb-team "iceblwdzs" \
  --wandb-project "qwen19b-slime-dapo" \
  --wandb-group "qwen19b-multinode-train" \
  --wandb-dir "/mnt/data/user01/19b100w/slime_runs/wandb"
```

`MODE=train` 使用 FP32 gradient accumulation/all-reduce；只有 smoke 为降低初始化显存使用 BF16 gradient reduce。正式配置精度更高，但显存压力也更大，不能把 smoke 成功等同于正式长序列一定成功。

launcher 的 checkpoint 规则：

- `CKPT_DIR/latest_checkpointed_iteration.txt` 不存在：从原 HF 权重开始。
- 该文件存在：自动从对应 slime checkpoint 续训。
- 要全新实验，使用新的 `EXPERIMENT_NAME` 或显式的新 `CKPT_DIR`；不要把不同拓扑/实验误指向旧 checkpoint。

训练日志写入 head 项目的 `logs/`，checkpoint/details/W&B 目录默认写到共享的 `OUTPUT_ROOT`。

## 8. 停止与故障排查

训练作业结束不等于 Ray daemon 自动停止。确认没有其他作业使用该集群后，在每台服务器分别执行：

```bash
SLIME_CONTAINER_NAME=slime-qwen19b-multinode \
bash scripts/docker_env.sh ray-stop
```

常见失败顺序：

1. worker 无法加入：检查 host network、私网路由、防火墙、`MASTER_ADDR/NODE_IP`。
2. NCCL/Gloo 超时：检查 `SOCKET_IFNAME` 是否为所有节点可通信的同类接口，以及代理变量是否绕过私网 IP。
3. preflight 签名不一致：重新同步项目、数据或模型，不能关闭检查硬跑。
4. actor 初始化 OOM：优先核对 TP/PP 和 SGLang memory fraction；正式模式保留 FP32 gradient accumulation 会比 smoke 占更多显存。
5. W&B 没有上传：确认每台容器都登录、`--wandb-mode online`，并检查 head 日志中的第一条 W&B 异常。

## 9. 验证边界

本交付版已在本机完成 shell/Python 静态检查、CPU 单测、原 6-GPU 单机配置 dry-run 和 2×8 GPU 多机配置 dry-run。单台服务器无法替代真实跨机网络、NCCL 和共享存储验证，因此接收方必须把第 6 节的 multi-node smoke 作为正式训练前的验收门槛。
