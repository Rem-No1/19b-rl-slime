# Qwen19B Big-Math DAPO with Slime

## 1. 这个项目是做什么的

本项目使用 [THUDM/slime](https://github.com/THUDM/slime) 对 Qwen3.5-MoE 19B 模型进行 DAPO/GRPO 数学强化学习训练，是原 verl 实验的 Slime 迁移版。

训练数据来自 Big-Math-RL-Verified 中两个低解题率 bucket：

- `solve_rate_0.00_0.05_with_system_prompt.jsonl`
- `solve_rate_0.05_0.10_with_system_prompt.jsonl`

验证集使用 AIME 2025。默认每 10 个 rollout step 验证一次，每 20 step 保存一次 checkpoint。

项目保留的主要训练语义：

- GRPO group mean/std advantage normalization。
- DAPO Clip-Higher，clip low/high 为 `0.20 / 0.28`。
- Token-level loss。
- Token-level TIS，权重截断上限为 `2.0`，随后进行 batch normalization。
- 使用原始正确率 `acc` 过滤全对或全错的采样组。
- 对 response 末尾 4096 token 施加线性 overlong penalty。
- KL reward/loss 均为 0。
- Qwen3.5-MoE rollout routing replay（R3）。
- 支持目标模型的逐 expert HF checkpoint，不修改原始 checkpoint 文件。
- 支持单机训练与 Docker + Ray 同构多机训练。
- 支持 AIME25 验证、checkpoint 自动续训、本地日志和 W&B。

当前已经完成 6×H100 80GB 单机 smoke；多机 launcher、2×8 GPU dry-run 和 Ray cluster preflight 已验证，但仍需接收方在真实多台服务器上完成 multi-node smoke。

## 2. 项目目录结构

```text
19b-rl-slime/
├── README.md
├── MULTINODE.md
├── run_dapo.sh
├── configs/
│   ├── multinode.env.example
│   └── tis_token_batch_normalized.yaml
├── data/
│   ├── big_math_dapo_train.jsonl
│   ├── aime25_val.jsonl
│   └── manifest.json
├── scripts/
│   ├── docker_env.sh
│   ├── ray_cluster.sh
│   ├── cluster_preflight.py
│   ├── preflight.py
│   ├── prepare_data.py
│   └── models/
│       └── qwen19b-100w.sh
├── slime_hooks/
│   ├── reward.py
│   └── qwen35_per_expert.py
├── tests/
└── third_party/
    └── slime/
```

各目录作用：

| 路径 | 作用 |
|---|---|
| `run_dapo.sh` | 训练总入口，组装 Slime、Megatron、SGLang、DAPO、验证和 W&B 参数 |
| `scripts/docker_env.sh` | 下载镜像，创建/进入容器，管理容器中的 Ray 节点 |
| `scripts/ray_cluster.sh` | 启动 Ray head、worker，查看状态或停止本机 Ray |
| `scripts/preflight.py` | 训练前检查模型结构、checkpoint、数据和并行拓扑 |
| `scripts/cluster_preflight.py` | 多机检查节点/GPU 数、文件一致性及共享输出目录 |
| `scripts/prepare_data.py` | 将原始 Big-Math/AIME25 messages JSONL 转成 Slime JSONL |
| `scripts/models/qwen19b-100w.sh` | Qwen3.5-MoE 19B 的 Megatron 模型结构参数 |
| `slime_hooks/reward.py` | 数学答案正确率、overlong shaping 和 dynamic filter |
| `slime_hooks/qwen35_per_expert.py` | 逐 expert HF checkpoint 兼容 loader |
| `configs/tis_token_batch_normalized.yaml` | Token TIS 配置 |
| `configs/multinode.env.example` | 2 台、每台 8 GPU 的多机配置模板 |
| `third_party/slime` | 固定 commit 的完整 Slime 框架源码 |
| `MULTINODE.md` | 多机 Docker、Ray、网络和训练交付手册 |

## 3. 镜像环境如何准备

### 3.1 前置条件

- Linux x86_64。
- NVIDIA GPU 和兼容驱动。
- 已安装 Docker 与 NVIDIA Container Toolkit。
- 模型、数据和输出目录能够挂载到容器。
- 多机训练时，各服务器之间有稳定的私网连接。

本项目使用的环境镜像已经发布到 [Docker Hub：`iceswallow/slime-qwen19b-dev`](https://hub.docker.com/r/iceswallow/slime-qwen19b-dev)。推荐使用固定版本标签：

```text
iceswallow/slime-qwen19b-dev:slime-f655e13
```

远端已验证 digest：

```text
sha256:2feaad36b157ee1f790f139aeb6d2669a466b914f28426f468df70b2324807a7
```

也提供 `iceswallow/slime-qwen19b-dev:latest`，但正式交付建议固定版本标签或 digest，避免以后 `latest` 更新导致环境变化。

这个镜像只包含 CUDA、PyTorch、Slime、Megatron-LM、SGLang、Ray 等运行环境，不包含本项目代码、模型、数据、checkpoint、日志或 W&B 凭据。项目代码由 GitHub 仓库提供，并在创建容器时通过 bind mount 挂载，因此修改代码不需要重新构建镜像。

项目内 Slime 固定 commit：

```text
f655e13d9b262748441e836983deaddfe4715e22
```

### 3.2 下载项目与 Slime 源码

```bash
git clone --recurse-submodules https://github.com/Rem-No1/19b-rl-slime.git
cd 19b-rl-slime
```

确认 `third_party/slime/train.py` 存在。如果发布时没有把 Slime 配置成 submodule，则手动拉取固定版本：

```bash
mkdir -p third_party
git clone https://github.com/THUDM/slime.git third_party/slime
git -C third_party/slime checkout f655e13d9b262748441e836983deaddfe4715e22
```

不要只升级 Slime、Megatron-LM 或 SGLang 中的一个组件，它们需要成套验证。

### 3.3 下载镜像并创建容器

可以先直接验证 Docker Hub 镜像能够下载：

```bash
docker pull iceswallow/slime-qwen19b-dev:slime-f655e13
```

然后在项目根目录固定镜像 digest，并创建开发容器：

```bash
export SLIME_DOCKER_IMAGE="iceswallow/slime-qwen19b-dev@sha256:2feaad36b157ee1f790f139aeb6d2669a466b914f28426f468df70b2324807a7"
export SLIME_CONTAINER_NAME="slime-qwen19b-dev"

bash scripts/docker_env.sh pull
bash scripts/docker_env.sh create
bash scripts/docker_env.sh status
bash scripts/docker_env.sh shell
```

默认行为：

- 容器名：`slime-qwen19b-dev`。
- 网络：host network。
- IPC：host IPC。
- 挂载当前项目到容器中的相同绝对路径。
- 挂载宿主机 `/mnt/data` 到容器 `/mnt/data`。
- 自动把 `third_party/slime` 注册为 editable Python package。
- 镜像内 Megatron-LM 路径为 `/root/Megatron-LM`。

`SLIME_DOCKER_IMAGE` 也可以设置成版本标签：

```bash
export SLIME_DOCKER_IMAGE="iceswallow/slime-qwen19b-dev:slime-f655e13"
```

使用 digest 的可复现性更强；使用版本标签更容易阅读。两者当前指向完全相同的镜像。

`bash scripts/docker_env.sh check` 会使用原实验的默认模型和数据路径做 preflight，因此应在模型和第 4 节数据准备完成后执行。数据路径经过调整时，使用第 5.1 节显式传参的 preflight 命令。

项目通过 bind mount 挂载，所以可以在宿主机直接修改代码，容器内会实时看到修改。切换 Slime 分支或重建源码目录后执行：

```bash
bash scripts/docker_env.sh install-source
```

停止或重新进入容器：

```bash
bash scripts/docker_env.sh stop
bash scripts/docker_env.sh start
bash scripts/docker_env.sh shell
```

多机时，每台服务器都要准备相同镜像和项目代码，并使用不同 hostname。完整步骤见 [MULTINODE.md](MULTINODE.md)。

## 4. 数据要什么格式，放到什么地方

### 4.1 原始数据格式

原始训练和验证数据均为 JSONL，每行至少包含 `messages`。最后一条消息必须是 gold assistant answer：

```json
{
  "messages": [
    {"role": "system", "content": "Please reason step by step and put the final answer in \\boxed{}."},
    {"role": "user", "content": "题目内容"},
    {"role": "assistant", "content": "推理过程与最终答案 \\boxed{42}"}
  ],
  "metadata": {
    "_sample_id": "sample-0001",
    "source": "dataset-name",
    "llama8b_solve_rate": 0.03,
    "_source_row": 1
  }
}
```

要求：

- `messages` 非空。
- 最后一条必须是 `assistant`，其内容会转换成 `label`。
- 最后一条之前至少有一条 `user`。
- prompt 部分只允许 `system` 和 `user`。
- 两个训练文件应使用相同 system prompt。
- `metadata` 可以缺少；脚本会为缺失字段提供默认值。

### 4.2 默认原始数据位置

默认转换脚本读取：

```text
/mnt/data/user01/LLMData/train/math/Big-Math-RL-Verified /rl_buckets_0.05/solve_rate_0.00_0.05_with_system_prompt.jsonl
/mnt/data/user01/LLMData/train/math/Big-Math-RL-Verified /rl_buckets_0.05/solve_rate_0.05_0.10_with_system_prompt.jsonl
/mnt/data/user01/LLMData/val/AIME25/AIME25.jsonl
```

注意 `Big-Math-RL-Verified ` 末尾确实包含一个空格，因此 shell 中必须给完整路径加引号。

如果数据放在其他位置，直接通过参数传入：

```bash
python3 scripts/prepare_data.py \
  --train-files \
    "/path/to/solve_rate_0.00_0.05_with_system_prompt.jsonl" \
    "/path/to/solve_rate_0.05_0.10_with_system_prompt.jsonl" \
  --val-file "/path/to/AIME25.jsonl" \
  --output-dir data
```

输出已经存在时，脚本默认拒绝覆盖；确定要重新生成时增加 `--force`。

### 4.3 转换后的 Slime 数据格式与位置

执行：

```bash
python3 scripts/prepare_data.py --output-dir data
```

生成文件：

| 文件 | 预期数量 | 用途 |
|---|---:|---|
| `data/big_math_dapo_train.jsonl` | 54,398 | 训练数据 |
| `data/aime25_val.jsonl` | 30 | AIME25 验证数据 |
| `data/manifest.json` | 1 | 输入 SHA256、行数、bucket 数量与 system prompt |

训练 JSONL 每行格式：

```json
{
  "prompt": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "label": "gold assistant answer",
  "metadata": {
    "source_name": "big_math_rl_verified",
    "sample_id": "sample-0001",
    "bucket": "solve_rate_0.00_0.05",
    "solve_rate": 0.03
  }
}
```

验证 JSONL 使用相同的 `prompt / label / metadata` 顶层字段，`metadata.source_name` 必须为 `aime25`。

默认 launcher 读取：

```text
TRAIN_DATA=<项目根目录>/data/big_math_dapo_train.jsonl
EVAL_DATA=<项目根目录>/data/aime25_val.jsonl
```

也可以在启动训练时通过 `TRAIN_DATA` 和 `EVAL_DATA` 指向其他绝对路径。

## 5. 训练启动命令（完整参数）

以下命令都应在 Docker 容器中的项目根目录执行。

### 5.1 训练前检查

```bash
MODEL_PATH=/mnt/data/user01/19b100w/Qwen19B-100w \
TRAIN_DATA="$PWD/data/big_math_dapo_train.jsonl" \
EVAL_DATA="$PWD/data/aime25_val.jsonl" \
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=6 \
TRAIN_TP=2 \
TRAIN_PP=1 \
TRAIN_CP=1 \
TRAIN_EP=2 \
ROLLOUT_TP=2 \
PRECHECK_ONLY=1 \
bash run_dapo.sh
```

这个命令不加载模型、不启动 Ray、不开始训练。

### 5.2 单机 smoke 完整命令

先设置公共输出目录：

```bash
export OUTPUT_ROOT=/mnt/data/user01/19b100w/slime_runs
export EXPERIMENT_NAME=qwen19b_slime_dapo_smoke
export CKPT_DIR="$OUTPUT_ROOT/checkpoints/$EXPERIMENT_NAME"
export DETAIL_DIR="$OUTPUT_ROOT/details/$EXPERIMENT_NAME"
export LOG_DIR="$PWD/logs"
```

启动 smoke：

```bash
MODE=smoke \
MODEL_PATH=/mnt/data/user01/19b100w/Qwen19B-100w \
TRAIN_DATA="$PWD/data/big_math_dapo_train.jsonl" \
EVAL_DATA="$PWD/data/aime25_val.jsonl" \
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=6 \
TRAIN_TP=2 \
TRAIN_PP=1 \
TRAIN_CP=1 \
TRAIN_EP=2 \
ROLLOUT_TP=2 \
ROLLOUT_BATCH_SIZE=3 \
ROLLOUT_N=2 \
MAX_PROMPT_LENGTH=512 \
MAX_RESPONSE_LENGTH=1024 \
MAX_TOKENS_PER_GPU=1536 \
TOTAL_STEPS=2 \
LR_WARMUP_STEPS=0 \
SAVE_INTERVAL=2 \
EVAL_INTERVAL="" \
EVAL_N=8 \
OVER_SAMPLING_BATCH_SIZE=6 \
DAPO_OVERLONG_BUFFER_LEN=0 \
DAPO_OVERLONG_PENALTY_FACTOR=1.0 \
FILTER_GROUPS=0 \
ACTOR_LR=1e-6 \
ROLLOUT_GPU_MEM_UTIL=0.45 \
ENABLE_R3=1 \
AUTO_PREPARE=0 \
PRECHECK_ONLY=0 \
DRY_RUN=0 \
RAY_CLUSTER_MODE=auto \
bash run_dapo.sh \
  --use-wandb \
  --wandb-mode online \
  --wandb-team "your-wandb-entity" \
  --wandb-project "qwen19b-slime-dapo" \
  --wandb-group "qwen19b-smoke" \
  --wandb-dir "$OUTPUT_ROOT/wandb"
```

如果不使用 W&B，删除 `--use-wandb` 到 `--wandb-dir` 这几行。

### 5.3 单机正式训练完整命令

为正式实验使用新的实验名：

```bash
export OUTPUT_ROOT=/mnt/data/user01/19b100w/slime_runs
export EXPERIMENT_NAME=qwen19b_slime_dapo_bigmath_0_010
export CKPT_DIR="$OUTPUT_ROOT/checkpoints/$EXPERIMENT_NAME"
export DETAIL_DIR="$OUTPUT_ROOT/details/$EXPERIMENT_NAME"
export LOG_DIR="$PWD/logs"
```

启动正式训练：

```bash
MODE=train \
MODEL_PATH=/mnt/data/user01/19b100w/Qwen19B-100w \
TRAIN_DATA="$PWD/data/big_math_dapo_train.jsonl" \
EVAL_DATA="$PWD/data/aime25_val.jsonl" \
CUDA_VISIBLE_DEVICES=0,1,2,4,5,6 \
ACTOR_NUM_NODES=1 \
ACTOR_NUM_GPUS_PER_NODE=6 \
TRAIN_TP=2 \
TRAIN_PP=1 \
TRAIN_CP=1 \
TRAIN_EP=2 \
ROLLOUT_TP=2 \
ROLLOUT_BATCH_SIZE=24 \
ROLLOUT_N=8 \
MAX_PROMPT_LENGTH=2048 \
MAX_RESPONSE_LENGTH=16384 \
MAX_TOKENS_PER_GPU=18432 \
TOTAL_STEPS=200 \
LR_WARMUP_STEPS=10 \
SAVE_INTERVAL=20 \
EVAL_INTERVAL=10 \
EVAL_N=8 \
OVER_SAMPLING_BATCH_SIZE=72 \
DAPO_OVERLONG_BUFFER_LEN=4096 \
DAPO_OVERLONG_PENALTY_FACTOR=1.0 \
FILTER_GROUPS=1 \
ACTOR_LR=1e-6 \
ROLLOUT_GPU_MEM_UTIL=0.45 \
ENABLE_R3=1 \
AUTO_PREPARE=0 \
PRECHECK_ONLY=0 \
DRY_RUN=0 \
RAY_CLUSTER_MODE=auto \
bash run_dapo.sh \
  --use-wandb \
  --wandb-mode online \
  --wandb-team "your-wandb-entity" \
  --wandb-project "qwen19b-slime-dapo" \
  --wandb-group "qwen19b-train" \
  --wandb-dir "$OUTPUT_ROOT/wandb"
```

正式模式会使用 FP32 gradient accumulation/all-reduce；smoke 使用 BF16 gradient reduce。

### 5.4 两台、每台 8 GPU 的启动方式

先在每台机器准备配置：

```bash
cp configs/multinode.env.example configs/multinode.env
```

把 `MASTER_ADDR`、`SOCKET_IFNAME`、模型路径和输出路径改成实际值。

head 节点：

```bash
set -a
source configs/multinode.env
set +a
export NODE_IP="$MASTER_ADDR"
export SLIME_CONTAINER_NAME=slime-qwen19b-multinode
bash scripts/docker_env.sh ray-head
```

每个 worker 节点，将 `NODE_IP` 改成本机私网 IP：

```bash
set -a
source configs/multinode.env
set +a
export NODE_IP=10.0.0.11
export SLIME_CONTAINER_NAME=slime-qwen19b-multinode
bash scripts/docker_env.sh ray-worker
```

最后只在 head 容器中提交一次。先设置输出目录：

```bash
# 在 head 宿主机先进入多机容器
SLIME_CONTAINER_NAME=slime-qwen19b-multinode \
bash scripts/docker_env.sh shell

# 以下命令在 head 容器内执行
set -a
source configs/multinode.env
set +a
export NODE_IP="$MASTER_ADDR"
export EXPERIMENT_NAME=qwen19b_multinode_dapo
export CKPT_DIR="$OUTPUT_ROOT/checkpoints/$EXPERIMENT_NAME"
export DETAIL_DIR="$OUTPUT_ROOT/details/$EXPERIMENT_NAME"
export LOG_DIR="$PWD/logs"
```

然后执行完整正式训练参数：

```bash
MODE=train \
ROLLOUT_BATCH_SIZE=24 \
ROLLOUT_N=8 \
MAX_PROMPT_LENGTH=2048 \
MAX_RESPONSE_LENGTH=16384 \
MAX_TOKENS_PER_GPU=18432 \
TOTAL_STEPS=200 \
LR_WARMUP_STEPS=10 \
SAVE_INTERVAL=20 \
EVAL_INTERVAL=10 \
EVAL_N=8 \
OVER_SAMPLING_BATCH_SIZE=72 \
DAPO_OVERLONG_BUFFER_LEN=4096 \
DAPO_OVERLONG_PENALTY_FACTOR=1.0 \
FILTER_GROUPS=1 \
ACTOR_LR=1e-6 \
ROLLOUT_GPU_MEM_UTIL=0.45 \
ENABLE_R3=1 \
AUTO_PREPARE=0 \
PRECHECK_ONLY=0 \
DRY_RUN=0 \
CLUSTER_PREFLIGHT=1 \
RAY_CLUSTER_MODE=existing \
bash run_dapo.sh \
  --use-wandb \
  --wandb-mode online \
  --wandb-team "your-wandb-entity" \
  --wandb-project "qwen19b-slime-dapo" \
  --wandb-group "qwen19b-multinode-train" \
  --wandb-dir "$OUTPUT_ROOT/wandb"
```

这里的 `ACTOR_NUM_NODES`、每节点 GPU 数、CUDA 卡号、TP/PP/CP/EP、`MASTER_ADDR`、模型路径和输出路径已经由 `configs/multinode.env` 提供。更详细的防火墙、共享存储和停止命令见 [MULTINODE.md](MULTINODE.md)。

## 6. 启动命令中各个参数的含义

### 6.1 路径、模式和输出参数

| 参数 | 含义 | smoke 默认值 | train 默认值 |
|---|---|---|---|
| `MODE` | `smoke` 为最小链路测试；`train` 为正式配置 | `smoke` | `train` |
| `MODEL_PATH` | Qwen19B HF checkpoint 目录 | 原实验模型路径 | 同左 |
| `TRAIN_DATA` | 转换后的训练 JSONL | `data/big_math_dapo_train.jsonl` | 同左 |
| `EVAL_DATA` | 转换后的 AIME25 JSONL | `data/aime25_val.jsonl` | 同左 |
| `OUTPUT_ROOT` | checkpoint/details/W&B 根目录 | `/mnt/data/user01/19b100w/slime_runs` | 同左 |
| `EXPERIMENT_NAME` | 实验名，并参与构造默认输出目录 | `qwen19b_slime_dapo_smoke` | `qwen19b_slime_dapo_bigmath_0_010` |
| `CKPT_DIR` | checkpoint 保存和自动恢复目录 | `$OUTPUT_ROOT/checkpoints/$EXPERIMENT_NAME` | 同左 |
| `DETAIL_DIR` | 每个 sample 的 rollout/reward details | `$OUTPUT_ROOT/details/$EXPERIMENT_NAME` | 同左 |
| `LOG_DIR` | head 节点文本日志目录 | `./logs` | `./logs` |
| `LOG_FILE` | 可选，显式指定本次日志文件 | 自动带时间戳 | 自动带时间戳 |

### 6.2 Batch、长度和训练步数

| 参数 | 含义 | smoke | train |
|---|---|---:|---:|
| `ROLLOUT_BATCH_SIZE` | 每个 rollout 使用的 prompt 数 | 3 | 24 |
| `ROLLOUT_N` | 每个 prompt 生成的 response 数 | 2 | 8 |
| `GLOBAL_BATCH_SIZE` | 脚本内部计算：`ROLLOUT_BATCH_SIZE × ROLLOUT_N` | 6 | 192 |
| `MAX_PROMPT_LENGTH` | prompt 最大 token 数 | 512 | 2048 |
| `MAX_RESPONSE_LENGTH` | response 最大 token 数 | 1024 | 16384 |
| `MAX_TOKENS_PER_GPU` | dynamic batch 每 GPU 最大 token 数 | 1536 | 18432 |
| `TOTAL_STEPS` | rollout/update 总步数 | 2 | 200 |
| `OVER_SAMPLING_BATCH_SIZE` | dynamic sampling 每轮候选 prompt 数 | 6 | 72 |
| `FILTER_GROUPS` | 是否丢弃全对/全错 group | 0 | 1 |
| `EVAL_INTERVAL` | 每多少 rollout step 做验证；空表示关闭周期验证 | 空 | 10 |
| `EVAL_N` | 每道验证题采样的 response 数 | 8 | 8 |
| `SAVE_INTERVAL` | 每多少 step 保存 checkpoint | 2 | 20 |

### 6.3 优化器、reward 和显存参数

| 参数 | 含义 | smoke | train |
|---|---|---:|---:|
| `ACTOR_LR` | actor Adam learning rate | 1e-6 | 1e-6 |
| `LR_WARMUP_STEPS` | learning-rate warmup step 数 | 0 | 10 |
| `DAPO_OVERLONG_BUFFER_LEN` | response 尾部线性超长惩罚区间；0 表示关闭 | 0 | 4096 |
| `DAPO_OVERLONG_PENALTY_FACTOR` | overlong penalty 系数 | 1.0 | 1.0 |
| `ROLLOUT_GPU_MEM_UTIL` | SGLang static memory fraction | 0.45 | 0.45 |
| `ENABLE_R3` | 是否启用 MoE rollout routing replay | 1 | 1 |

脚本固定使用 Adam、weight decay `0.1`、betas `(0.9, 0.999)`、gradient clip `1.0`、optimizer CPU offload、constant LR，以及 full recompute。

### 6.4 并行与 GPU 参数

| 参数 | 含义 | 单机默认 |
|---|---|---:|
| `CUDA_VISIBLE_DEVICES` | 每个节点向 Ray/训练暴露的 GPU 编号 | `0,1,2,4,5,6` |
| `ACTOR_NUM_NODES` | actor 使用的节点数 | 1 |
| `ACTOR_NUM_GPUS_PER_NODE` | 每节点 actor GPU 数，必须等于可见 GPU 数 | 6 |
| `TRAIN_TP` | Megatron tensor parallel size | 2 |
| `TRAIN_PP` | Megatron pipeline parallel size | 1 |
| `TRAIN_CP` | Megatron context parallel size | 1 |
| `TRAIN_EP` | Megatron expert parallel size | 2 |
| `ROLLOUT_TP` | 每个 SGLang rollout engine 使用的 GPU 数 | 2 |

拓扑必须满足脚本的 preflight 约束，例如 `TP × PP × CP`、`EP × PP` 和 rollout TP 都要与 actor 总 GPU 数兼容；TP 还必须兼容该模型的 16 attention heads 和 2 query groups。

### 6.5 Ray、多机和控制参数

| 参数 | 含义 | 默认值 |
|---|---|---|
| `RAY_CLUSTER_MODE` | `auto` 单机自动启动 Ray；`existing` 使用已启动集群 | 单机 `auto`，多机 `existing` |
| `MASTER_ADDR` | Ray head 私网 IP | 单机 `127.0.0.1` |
| `NODE_IP` | 当前节点私网 IP | 默认为 `MASTER_ADDR` |
| `RAY_PORT` | Ray GCS 端口 | 6379 |
| `RAY_DASHBOARD_PORT` | Ray Jobs/dashboard 端口 | 8265 |
| `RAY_DASHBOARD_ADDRESS` | `ray job submit` 使用的 HTTP 地址 | `http://127.0.0.1:8265` |
| `SOCKET_IFNAME` | 多机 NCCL/Gloo 使用的网卡，如 `ib0` | 无 |
| `CLUSTER_PREFLIGHT` | 多机提交前是否检查所有节点 | 1 |
| `REUSE_RAY` | 单机 auto 模式是否允许复用当前可见 Ray | 0 |
| `RESTART_RAY` | `ray_cluster.sh` 是否先停止本机旧 Ray | 0 |
| `AUTO_PREPARE` | 数据不存在时是否自动运行默认数据转换 | 1 |
| `PRECHECK_ONLY` | 只做 preflight 后退出 | 0 |
| `DRY_RUN` | 只打印完整 Slime 命令，不启动训练 | 0 |

### 6.6 W&B 参数

| 参数 | 含义 |
|---|---|
| `--use-wandb` | 启用 W&B |
| `--wandb-mode online` | 在线上传；也可使用 `offline` 或 `disabled` |
| `--wandb-team` | W&B entity/team 名称 |
| `--wandb-project` | W&B project 名称 |
| `--wandb-group` | 本次 run 的 group/name 基础值 |
| `--wandb-dir` | 本地 W&B 文件目录，建议放在持久化存储 |
| `--wandb-run-id` | 可选，显式指定 run ID，用于恢复同一个 W&B run |
| `--disable-wandb-random-suffix` | 可选，不给 group/run name 添加随机后缀 |

容器中先执行 `wandb login`。不要把 API key 直接写进 README 或脚本；多机时每个持久容器都要登录，或由 secret manager 注入凭据。

### 6.7 脚本固定生成的关键 DAPO/Slime 参数

| Slime 参数 | 当前值 | 含义 |
|---|---:|---|
| `--advantage-estimator` | `grpo` | 使用 GRPO advantage |
| `--eps-clip / --eps-clip-high` | `0.2 / 0.28` | DAPO asymmetric clipping |
| `--kl-coef / --kl-loss-coef` | `0 / 0` | 不使用 KL reward/loss |
| `--entropy-coef` | `0` | 不增加 entropy loss |
| `--use-tis --tis-clip` | `true / 2.0` | Token-level truncated importance sampling |
| `--calculate-per-token-loss` | true | Token-level loss aggregation |
| `--apply-chat-template-kwargs` | `enable_thinking=true` | 使用模型 thinking chat template |
| `--rollout-temperature / top-p / top-k` | `1.0 / 1.0 / -1` | Rollout 采样参数 |
| `--eval-temperature / top-p` | `1.0 / 1.0` | AIME25 采样参数 |
| `--grad-reduce-in-bf16` | smoke only | 降低 smoke 初始化显存 |
| `--accumulate-allreduce-grads-in-fp32` | train only | 正式训练使用更高精度梯度归约 |

`run_dapo.sh` 后面追加的其他参数会原样传给 `third_party/slime/train.py`。

## 7. 其他可能需要注意的地方

### 7.1 Checkpoint 自动恢复

- 如果 `CKPT_DIR/latest_checkpointed_iteration.txt` 不存在，launcher 从 `MODEL_PATH` 的原始 HF 权重初始化。
- 如果 tracker 存在，launcher 自动从 `CKPT_DIR` 续训。
- 要重新开始，使用新的 `EXPERIMENT_NAME` 或新的 `CKPT_DIR`。
- 不要让不同模型或不兼容实验误用同一 checkpoint 目录。

### 7.2 Smoke 与正式训练精度不同

- `MODE=smoke` 使用 BF16 gradient reduce，以避免 SGLang colocate 初始化时显存不足。
- `MODE=train` 使用 FP32 gradient accumulation/all-reduce，精度更高，但显存压力明显更大。
- 6 卡 smoke 成功不代表 16384 response length 的正式配置一定不会 OOM。正式训练前应做逐步放大测试。

### 7.3 多机存储和网络

- 每台机器必须使用相同镜像、Slime commit 和项目代码。
- 项目、模型和数据必须使用相同绝对路径。
- `OUTPUT_ROOT` 必须是真正的共享文件系统，不能只是各节点恰好同名的本地目录。
- 每台服务器 hostname 必须不同。
- `SOCKET_IFNAME` 要设置为所有节点实际互通的 IB/Ethernet 网卡。
- Ray GCS/dashboard 不应暴露到公网。
- 多机 preflight 会在每个 Ray NodeID 上检查文件签名、模型 shard、Slime revision、GPU 数和共享目录临时文件。

### 7.4 Ray 进程管理

- launcher 不会使用 `pkill` 清理其他用户的 Python/Ray 任务。
- 单机发现已有 Ray 时默认报错；只有确认它属于本实验时才设置 `REUSE_RAY=1`。
- `scripts/ray_cluster.sh` 只有在显式设置 `RESTART_RAY=1` 时才停止本机旧 Ray。
- 训练结束后 Ray daemon 不一定自动退出；确认无其他作业使用后，在每个节点执行 `bash scripts/docker_env.sh ray-stop`。

### 7.5 数据、模型与凭据不要误传 GitHub

- 不要上传模型权重、checkpoint、`.netrc`、W&B key 或其他密钥。
- 默认容器只额外挂载 `/mnt/data`；项目目录以外的模型、数据和输出应放在 `/mnt/data` 下，或修改 `scripts/docker_env.sh` 增加对应 bind mount。
- 原始数据和转换后数据能否公开取决于数据许可证；不确定时只提交转换脚本，不提交 `data/*.jsonl`。
- `logs/` 应保持在 `.gitignore` 中。
- `third_party/slime` 是独立 Git 仓库。上传主仓库前应配置为正式 submodule，或者排除后让使用者按固定 commit 下载；不要提交没有 `.gitmodules` 的嵌入式 gitlink。

### 7.6 测试与问题定位

CPU 测试和静态检查：

```bash
PYTHONDONTWRITEBYTECODE=1 \
python3 -m pytest tests -q -p no:cacheprovider

ruff check scripts/preflight.py scripts/cluster_preflight.py tests
bash -n run_dapo.sh scripts/docker_env.sh scripts/ray_cluster.sh
```

问题定位顺序：

1. 先查看 `logs/<experiment>_<timestamp>.log` 中的第一条 traceback。
2. 再检查 `ray status` 和 GPU 显存占用。
3. 多机 worker 加入失败时检查 IP、host network 和防火墙。
4. NCCL/Gloo 超时时检查 `SOCKET_IFNAME` 和私网代理绕过变量。
5. 初始化 OOM 时优先调整 TP/PP、长度和 `ROLLOUT_GPU_MEM_UTIL`，不要直接降低正式训练精度。

### 7.7 当前验证边界

- 已完成 15 个 CPU tests、Ruff、Bash syntax 和真实单机 smoke。
- 已完成 2×8 GPU 命令生成与本地 Ray cluster preflight。
- 尚未在真实两台物理服务器上完成 NCCL multi-node smoke；接收方必须先按 [MULTINODE.md](MULTINODE.md) 验收，再运行正式多机训练。
