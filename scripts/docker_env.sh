#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONTAINER_NAME="${SLIME_CONTAINER_NAME:-slime-qwen19b-dev}"
IMAGE="${SLIME_DOCKER_IMAGE:-slimerl/slime@sha256:2feaad36b157ee1f790f139aeb6d2669a466b914f28426f468df70b2324807a7}"
SLIME_SOURCE="$PROJECT_ROOT/third_party/slime"
NETWORK_MODE="${SLIME_DOCKER_NETWORK:-host}"
CONTAINER_HOSTNAME="${SLIME_CONTAINER_HOSTNAME:-$(hostname -s)}"

usage() {
  echo "Usage: $0 {pull|create|start|install-source|shell|check|status|stop|ray-head|ray-worker|ray-status|ray-stop}"
}

container_exists() {
  docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1
}

container_running() {
  [[ "$(docker container inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null || true)" == "true" ]]
}

ensure_running() {
  if ! container_exists; then
    echo "[error] container '$CONTAINER_NAME' does not exist; run: $0 create" >&2
    exit 2
  fi
  if ! container_running; then
    docker start "$CONTAINER_NAME" >/dev/null
  fi
}

install_source() {
  ensure_running
  if [[ ! -f "$SLIME_SOURCE/train.py" ]]; then
    echo "[error] slime source not found: $SLIME_SOURCE" >&2
    exit 2
  fi
  docker exec "$CONTAINER_NAME" \
    python3 -m pip install -e "$SLIME_SOURCE" --no-deps
}

run_ray_action() {
  local action="$1"
  ensure_running
  local -a command=(docker exec -w "$PROJECT_ROOT")
  local key
  for key in MASTER_ADDR NODE_IP RAY_PORT RAY_DASHBOARD_PORT GPUS_PER_NODE \
    ACTOR_NUM_GPUS_PER_NODE CUDA_VISIBLE_DEVICES SOCKET_IFNAME \
    GLOO_SOCKET_IFNAME NCCL_SOCKET_IFNAME TP_SOCKET_IFNAME \
    NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME RESTART_RAY; do
    if [[ -n "${!key:-}" ]]; then
      command+=(-e "$key=${!key}")
    fi
  done
  command+=("$CONTAINER_NAME" bash scripts/ray_cluster.sh "$action")
  "${command[@]}"
}

case "${1:-}" in
  pull)
    docker pull "$IMAGE"
    ;;
  create)
    if container_exists; then
      echo "[ok] container already exists: $CONTAINER_NAME"
      actual_network="$(docker container inspect -f '{{.HostConfig.NetworkMode}}' "$CONTAINER_NAME")"
      if [[ "$actual_network" != "$NETWORK_MODE" ]]; then
        echo "[warning] existing container network=$actual_network, requested=$NETWORK_MODE" >&2
        echo "[warning] multi-node use requires host or another cross-host routable network" >&2
      fi
      ensure_running
      exit 0
    fi
    docker run -d \
      --name "$CONTAINER_NAME" \
      --hostname "$CONTAINER_HOSTNAME" \
      --network "$NETWORK_MODE" \
      --gpus all \
      --ipc=host \
      --shm-size=16g \
      --cap-add SYS_NICE \
      --ulimit memlock=-1 \
      --ulimit stack=67108864 \
      -v "$PROJECT_ROOT:$PROJECT_ROOT" \
      -v /mnt/data:/mnt/data \
      -w "$PROJECT_ROOT" \
      "$IMAGE" \
      sleep infinity
    install_source
    ;;
  start)
    ensure_running
    echo "[ok] running: $CONTAINER_NAME"
    ;;
  install-source)
    install_source
    ;;
  shell)
    ensure_running
    exec docker exec -it -w "$PROJECT_ROOT" "$CONTAINER_NAME" bash
    ;;
  check)
    ensure_running
    docker exec -w "$PROJECT_ROOT" "$CONTAINER_NAME" bash -lc \
      'MEGATRON_ROOT=/root/Megatron-LM PRECHECK_ONLY=1 bash run_dapo.sh'
    docker exec "$CONTAINER_NAME" python3 -c \
      'import slime; print("[source ok]", slime.__file__)'
    ;;
  status)
    docker ps -a --filter "name=^/${CONTAINER_NAME}$" \
      --format '{{.ID}} {{.Names}} {{.Status}} {{.Image}}'
    if container_exists; then
      docker container inspect -f 'hostname={{.Config.Hostname}} network={{.HostConfig.NetworkMode}} ipc={{.HostConfig.IpcMode}}' "$CONTAINER_NAME"
    fi
    ;;
  ray-head)
    run_ray_action head
    ;;
  ray-worker)
    run_ray_action worker
    ;;
  ray-status)
    run_ray_action status
    ;;
  ray-stop)
    run_ray_action stop
    ;;
  stop)
    if container_running; then
      docker stop "$CONTAINER_NAME"
    else
      echo "[ok] container is already stopped or absent: $CONTAINER_NAME"
    fi
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
