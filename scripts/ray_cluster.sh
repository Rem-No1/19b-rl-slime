#!/usr/bin/env bash
set -euo pipefail

ACTION="${1:-}"
RAY_BIN="${RAY_BIN:-ray}"
MASTER_ADDR="${MASTER_ADDR:-}"
NODE_IP="${NODE_IP:-}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${ACTOR_NUM_GPUS_PER_NODE:-8}}"

usage() {
  cat <<'EOF'
Usage: scripts/ray_cluster.sh {head|worker|status|stop}

Required for head:   MASTER_ADDR=<head-ip> NODE_IP=<head-ip>
Required for worker: MASTER_ADDR=<head-ip> NODE_IP=<this-node-ip>
Optional: GPUS_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,...,7 RESTART_RAY=1
EOF
}

require_network_config() {
  if [[ -z "$MASTER_ADDR" || -z "$NODE_IP" ]]; then
    echo "[error] MASTER_ADDR and NODE_IP are required" >&2
    exit 2
  fi
  if [[ "$MASTER_ADDR" == "127.0.0.1" || "$NODE_IP" == "127.0.0.1" ]]; then
    echo "[error] loopback addresses cannot be used for a multi-node Ray cluster" >&2
    exit 2
  fi
  if [[ "$GPUS_PER_NODE" -le 0 ]]; then
    echo "[error] GPUS_PER_NODE must be positive" >&2
    exit 2
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a visible_gpus <<<"$CUDA_VISIBLE_DEVICES"
    if [[ "${#visible_gpus[@]}" -ne "$GPUS_PER_NODE" ]]; then
      echo "[error] CUDA_VISIBLE_DEVICES exposes ${#visible_gpus[@]} GPUs, expected $GPUS_PER_NODE" >&2
      exit 2
    fi
  fi
}

prepare_network_env() {
  local bypass="localhost,127.0.0.1,0.0.0.0,$MASTER_ADDR,$NODE_IP"
  export no_proxy="${no_proxy:+$no_proxy,}$bypass"
  export NO_PROXY="${NO_PROXY:+$NO_PROXY,}$bypass"
  if [[ -n "${SOCKET_IFNAME:-}" ]]; then
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-$SOCKET_IFNAME}"
    export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-$SOCKET_IFNAME}"
    export TP_SOCKET_IFNAME="${TP_SOCKET_IFNAME:-$SOCKET_IFNAME}"
    export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME="${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-$SOCKET_IFNAME}"
  fi
}

maybe_stop_local_ray() {
  if [[ "${RESTART_RAY:-0}" == "1" ]]; then
    "$RAY_BIN" stop --force
  fi
}

case "$ACTION" in
  head)
    require_network_config
    if [[ "$NODE_IP" != "$MASTER_ADDR" ]]; then
      echo "[error] Ray head requires NODE_IP to equal MASTER_ADDR" >&2
      exit 2
    fi
    prepare_network_env
    maybe_stop_local_ray
    "$RAY_BIN" start --head \
      --node-ip-address "$NODE_IP" \
      --port "$RAY_PORT" \
      --num-gpus "$GPUS_PER_NODE" \
      --disable-usage-stats \
      --dashboard-host 0.0.0.0 \
      --dashboard-port "$RAY_DASHBOARD_PORT"
    echo "[ok] Ray head: $MASTER_ADDR:$RAY_PORT, dashboard: http://$MASTER_ADDR:$RAY_DASHBOARD_PORT"
    ;;
  worker)
    require_network_config
    if [[ "$NODE_IP" == "$MASTER_ADDR" ]]; then
      echo "[error] worker NODE_IP must differ from MASTER_ADDR" >&2
      exit 2
    fi
    prepare_network_env
    maybe_stop_local_ray
    "$RAY_BIN" start \
      --address "$MASTER_ADDR:$RAY_PORT" \
      --node-ip-address "$NODE_IP" \
      --num-gpus "$GPUS_PER_NODE" \
      --disable-usage-stats
    echo "[ok] Ray worker $NODE_IP joined $MASTER_ADDR:$RAY_PORT"
    ;;
  status)
    if [[ -z "$MASTER_ADDR" ]]; then
      echo "[error] MASTER_ADDR is required for status" >&2
      exit 2
    fi
    RAY_ADDRESS="$MASTER_ADDR:$RAY_PORT" "$RAY_BIN" status
    ;;
  stop)
    "$RAY_BIN" stop --force
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
