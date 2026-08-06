#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLIME_ROOT="${SLIME_ROOT:-$PROJECT_ROOT/third_party/slime}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
PYTHON_BIN="${SLIME_PYTHON:-python3}"
RAY_BIN="${RAY_BIN:-ray}"
MODEL_PATH="${MODEL_PATH:-/mnt/data/user01/19b100w/Qwen19B-100w}"
TRAIN_DATA="${TRAIN_DATA:-$PROJECT_ROOT/data/big_math_dapo_train.jsonl}"
EVAL_DATA="${EVAL_DATA:-$PROJECT_ROOT/data/aime25_val.jsonl}"
MODE="${MODE:-train}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,4,5,6}"
export CUDA_VISIBLE_DEVICES PYTHONUNBUFFERED=1
IFS=',' read -r -a VISIBLE_GPUS <<<"$CUDA_VISIBLE_DEVICES"
VISIBLE_GPU_COUNT="${#VISIBLE_GPUS[@]}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-${NGPUS_PER_NODE:-$VISIBLE_GPU_COUNT}}"
NGPUS_PER_NODE="$ACTOR_NUM_GPUS_PER_NODE" # Backward-compatible alias.
TOTAL_ACTOR_GPUS=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE))
TRAIN_TP="${TRAIN_TP:-2}"
TRAIN_PP="${TRAIN_PP:-1}"
TRAIN_CP="${TRAIN_CP:-1}"
TRAIN_EP="${TRAIN_EP:-2}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"

case "$MODE" in
  smoke)
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-3}"
    ROLLOUT_N="${ROLLOUT_N:-2}"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
    TOTAL_STEPS="${TOTAL_STEPS:-2}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-2}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-}"
    OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-6}"
    DAPO_OVERLONG_BUFFER_LEN="${DAPO_OVERLONG_BUFFER_LEN:-0}"
    FILTER_GROUPS="${FILTER_GROUPS:-0}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen19b_slime_dapo_smoke}"
    ;;
  train)
    ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-24}"
    ROLLOUT_N="${ROLLOUT_N:-8}"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
    TOTAL_STEPS="${TOTAL_STEPS:-200}"
    LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-10}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
    EVAL_INTERVAL="${EVAL_INTERVAL:-10}"
    OVER_SAMPLING_BATCH_SIZE="${OVER_SAMPLING_BATCH_SIZE:-72}"
    DAPO_OVERLONG_BUFFER_LEN="${DAPO_OVERLONG_BUFFER_LEN:-4096}"
    FILTER_GROUPS="${FILTER_GROUPS:-1}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen19b_slime_dapo_bigmath_0_010}"
    ;;
  *)
    echo "[error] MODE must be smoke or train, got: $MODE" >&2
    exit 2
    ;;
esac
export DAPO_OVERLONG_BUFFER_LEN
export DAPO_OVERLONG_PENALTY_FACTOR="${DAPO_OVERLONG_PENALTY_FACTOR:-1.0}"

GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * ROLLOUT_N))
MAX_CONTEXT_LENGTH=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$MAX_CONTEXT_LENGTH}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.45}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/data/user01/19b100w/slime_runs}"
CKPT_DIR="${CKPT_DIR:-$OUTPUT_ROOT/checkpoints/$EXPERIMENT_NAME}"
DETAIL_DIR="${DETAIL_DIR:-$OUTPUT_ROOT/details/$EXPERIMENT_NAME}"
LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
AUTO_PREPARE="${AUTO_PREPARE:-1}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
ENABLE_R3="${ENABLE_R3:-1}"

if [[ "$ACTOR_NUM_NODES" -le 0 || "$ACTOR_NUM_GPUS_PER_NODE" -le 0 ]]; then
  echo "[error] ACTOR_NUM_NODES and ACTOR_NUM_GPUS_PER_NODE must be positive" >&2
  exit 2
fi
if [[ "$VISIBLE_GPU_COUNT" -ne "$ACTOR_NUM_GPUS_PER_NODE" ]]; then
  echo "[error] CUDA_VISIBLE_DEVICES exposes $VISIBLE_GPU_COUNT GPUs, but ACTOR_NUM_GPUS_PER_NODE=$ACTOR_NUM_GPUS_PER_NODE" >&2
  exit 2
fi

if [[ "$AUTO_PREPARE" == "1" && (! -f "$TRAIN_DATA" || ! -f "$EVAL_DATA") ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/prepare_data.py" --output-dir "$PROJECT_ROOT/data"
fi

"$PYTHON_BIN" "$PROJECT_ROOT/scripts/preflight.py" \
  --model "$MODEL_PATH" \
  --train-data "$TRAIN_DATA" \
  --eval-data "$EVAL_DATA" \
  --slime-root "$SLIME_ROOT" \
  --ngpus "$TOTAL_ACTOR_GPUS" \
  --train-tp "$TRAIN_TP" \
  --train-pp "$TRAIN_PP" \
  --train-cp "$TRAIN_CP" \
  --train-ep "$TRAIN_EP" \
  --rollout-tp "$ROLLOUT_TP"

if [[ "$PRECHECK_ONLY" == "1" ]]; then
  exit 0
fi

source "$PROJECT_ROOT/scripts/models/qwen19b-100w.sh"

if [[ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]]; then
  LOAD_PATH="$CKPT_DIR"
else
  LOAD_PATH="$MODEL_PATH"
fi

CKPT_ARGS=(
  --hf-checkpoint "$MODEL_PATH"
  --load "$LOAD_PATH"
  --save "$CKPT_DIR"
  --save-interval "$SAVE_INTERVAL"
)

ROLLOUT_ARGS=(
  --prompt-data "$TRAIN_DATA"
  --input-key prompt
  --label-key label
  --metadata-key metadata
  --apply-chat-template
  --apply-chat-template-kwargs '{"enable_thinking": true}'
  --rollout-shuffle
  --custom-rm-path slime_hooks.reward.compute_reward
  --reward-key score
  --eval-reward-key acc
  --over-sampling-batch-size "$OVER_SAMPLING_BATCH_SIZE"
  --num-rollout "$TOTAL_STEPS"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$ROLLOUT_N"
  --num-steps-per-rollout 1
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --rollout-max-prompt-len "$MAX_PROMPT_LENGTH"
  --rollout-max-response-len "$MAX_RESPONSE_LENGTH"
  --rollout-max-context-len "$MAX_CONTEXT_LENGTH"
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --rollout-top-k -1
  --balance-data
)
if [[ "$FILTER_GROUPS" == "1" ]]; then
  ROLLOUT_ARGS+=(--dynamic-sampling-filter-path slime_hooks.reward.check_nonzero_acc_std)
fi

EVAL_ARGS=(
  --eval-prompt-data aime25 "$EVAL_DATA"
  --eval-input-key prompt
  --eval-label-key label
  --n-samples-per-eval-prompt "${EVAL_N:-8}"
  --eval-max-response-len "$MAX_RESPONSE_LENGTH"
  --eval-temperature 1.0
  --eval-top-p 1.0
)
if [[ -n "$EVAL_INTERVAL" ]]; then
  EVAL_ARGS+=(--eval-interval "$EVAL_INTERVAL")
else
  EVAL_ARGS+=(--skip-eval-before-train)
fi

PERF_ARGS=(
  --tensor-model-parallel-size "$TRAIN_TP"
  --sequence-parallel
  --pipeline-model-parallel-size "$TRAIN_PP"
  --context-parallel-size "$TRAIN_CP"
  --expert-model-parallel-size "$TRAIN_EP"
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --calculate-per-token-loss
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
  --log-probs-max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-coef 0.0
  --kl-loss-coef 0.0
  --entropy-coef 0.0
  --eps-clip 0.2
  --eps-clip-high 0.28
  --use-tis
  --tis-clip 2.0
  --custom-config-path "$PROJECT_ROOT/configs/tis_token_batch_normalized.yaml"
  --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "$ACTOR_LR"
  --lr-decay-style constant
  --lr-warmup-iters "$LR_WARMUP_STEPS"
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.999
  --clip-grad 1.0
  --optimizer-cpu-offload
  --overlap-cpu-optimizer-d2h-h2d
  --use-precision-aware-optimizer
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine "$ROLLOUT_TP"
  --sglang-mem-fraction-static "$ROLLOUT_GPU_MEM_UTIL"
  --sglang-mamba-scheduler-strategy extra_buffer
)

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
  --attention-backend flash
  --custom-megatron-init-path slime_hooks.qwen35_per_expert.install_qwen35_per_expert_loader
  --dump-details "$DETAIL_DIR"
)
if [[ "$MODE" == "smoke" ]]; then
  # Keep the six-GPU smoke test lightweight enough to initialize the actor.
  MISC_ARGS+=(--grad-reduce-in-bf16)
else
  # Preserve higher-precision gradient accumulation/reduction for real training.
  MISC_ARGS+=(--accumulate-allreduce-grads-in-fp32)
fi
if [[ "$ENABLE_R3" == "1" ]]; then
  MISC_ARGS+=(--use-rollout-routing-replay)
fi

TRAIN_CMD=(
  "$PYTHON_BIN" "$SLIME_ROOT/train.py"
  --actor-num-nodes "$ACTOR_NUM_NODES"
  --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE"
  --num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE"
  --colocate
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${GRPO_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${EVAL_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
  "$@"
)

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${TRAIN_CMD[@]}"
  printf '\n'
  exit 0
fi

mkdir -p "$CKPT_DIR" "$DETAIL_DIR" "$LOG_DIR"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONPATH="$PROJECT_ROOT:$SLIME_ROOT:$MEGATRON_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
NODE_IP="${NODE_IP:-$MASTER_ADDR}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
RAY_DASHBOARD_ADDRESS="${RAY_DASHBOARD_ADDRESS:-http://127.0.0.1:$RAY_DASHBOARD_PORT}"
CLUSTER_PREFLIGHT="${CLUSTER_PREFLIGHT:-1}"
NETWORK_BYPASS="localhost,127.0.0.1,0.0.0.0,$MASTER_ADDR,$NODE_IP"
export no_proxy="${no_proxy:+$no_proxy,}$NETWORK_BYPASS"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$NETWORK_BYPASS"
if [[ -n "${SOCKET_IFNAME:-}" ]]; then
  export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$SOCKET_IFNAME}"
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$SOCKET_IFNAME}"
  export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-$SOCKET_IFNAME}"
  export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-$SOCKET_IFNAME}"
fi
if [[ -z "${RAY_CLUSTER_MODE:-}" ]]; then
  if [[ "$ACTOR_NUM_NODES" -eq 1 ]]; then
    RAY_CLUSTER_MODE="auto"
  else
    RAY_CLUSTER_MODE="existing"
  fi
fi

case "$RAY_CLUSTER_MODE" in
  auto)
    if [[ "$ACTOR_NUM_NODES" -ne 1 ]]; then
      echo "[error] RAY_CLUSTER_MODE=auto is only supported for one node; start head/workers first for multi-node" >&2
      exit 2
    fi
    if "$RAY_BIN" status >/dev/null 2>&1; then
      if [[ "${REUSE_RAY:-0}" != "1" ]]; then
        echo "[error] an existing Ray cluster is visible; set REUSE_RAY=1 only if this job should use it" >&2
        exit 2
      fi
    else
      "$RAY_BIN" start --head \
        --node-ip-address "$NODE_IP" \
        --port "$RAY_PORT" \
        --num-gpus "$ACTOR_NUM_GPUS_PER_NODE" \
        --disable-usage-stats \
        --dashboard-host 0.0.0.0 \
        --dashboard-port "$RAY_DASHBOARD_PORT"
    fi
    ;;
  existing)
    export RAY_ADDRESS="${RAY_ADDRESS:-$MASTER_ADDR:$RAY_PORT}"
    if ! "$RAY_BIN" status >/dev/null 2>&1; then
      echo "[error] Ray cluster is not reachable at $RAY_ADDRESS; start scripts/ray_cluster.sh on head/workers first" >&2
      exit 2
    fi
    ;;
  *)
    echo "[error] RAY_CLUSTER_MODE must be auto or existing, got: $RAY_CLUSTER_MODE" >&2
    exit 2
    ;;
esac

if [[ "$ACTOR_NUM_NODES" -gt 1 && "$CLUSTER_PREFLIGHT" == "1" ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/cluster_preflight.py" \
    --address "$MASTER_ADDR:$RAY_PORT" \
    --expected-nodes "$ACTOR_NUM_NODES" \
    --gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
    --model "$MODEL_PATH" \
    --train-data "$TRAIN_DATA" \
    --eval-data "$EVAL_DATA" \
    --project-root "$PROJECT_ROOT" \
    --slime-root "$SLIME_ROOT" \
    --output-root "$OUTPUT_ROOT"
fi

export MASTER_ADDR
RUNTIME_ENV_JSON="$($PYTHON_BIN -c 'import json, os; keys=("PYTHONPATH","CUDA_VISIBLE_DEVICES","CUDA_DEVICE_MAX_CONNECTIONS","DAPO_OVERLONG_BUFFER_LEN","DAPO_OVERLONG_PENALTY_FACTOR","MASTER_ADDR","GLOO_SOCKET_IFNAME","NCCL_SOCKET_IFNAME","TP_SOCKET_IFNAME","NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME","no_proxy","NO_PROXY"); print(json.dumps({"env_vars": {k: os.environ[k] for k in keys if os.environ.get(k)}}))')"
"$RAY_BIN" job submit \
  --address "$RAY_DASHBOARD_ADDRESS" \
  --runtime-env-json "$RUNTIME_ENV_JSON" \
  -- "${TRAIN_CMD[@]}"
