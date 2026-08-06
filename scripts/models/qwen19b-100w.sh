#!/usr/bin/env bash

NLAYERS=40
MOE_LAYER_FREQ="[$(printf '1,%.0s' $(seq 1 $((NLAYERS - 1))))1]"

MODEL_ARGS=(
  --spec slime_plugins.models.qwen3_5 get_qwen3_5_spec

  --disable-bias-linear
  --qk-layernorm
  --group-query-attention
  --num-attention-heads 16
  --num-query-groups 2
  --kv-channels 256
  --num-layers 40
  --hidden-size 2048
  --ffn-hidden-size 512
  --use-gated-attention

  --normalization RMSNorm
  --apply-layernorm-1p
  --position-embedding-type rope
  --norm-epsilon 1e-6
  --rotary-percent 0.25
  --swiglu
  --untie-embeddings-and-output-weights
  --vocab-size 248320
  --rotary-base 10000000

  --moe-ffn-hidden-size 512
  --moe-shared-expert-intermediate-size 512
  --moe-router-score-function softmax
  --moe-token-dispatcher-type alltoall
  --moe-router-topk 8
  --moe-layer-freq "$MOE_LAYER_FREQ"
  --num-experts 128
  --moe-grouped-gemm
  --moe-token-drop-policy probs
  --moe-router-dtype fp32
  --moe-permute-fusion
  --moe-router-load-balancing-type aux_loss
  --moe-aux-loss-coeff "${ROUTER_AUX_LOSS_COEF:-0.001}"
  --moe-z-loss-coeff "${ROUTER_Z_LOSS_COEF:-0.0001}"

  --attention-output-gate
  --moe-shared-expert-gate
)

