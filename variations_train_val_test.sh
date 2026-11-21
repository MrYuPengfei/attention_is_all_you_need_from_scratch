#!/usr/bin/env bash
# 单次只启动一个论文变体，不循环跑全部实验。
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-$HOME/pytorch-cuda13.4-venv/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="$SCRIPT_DIR/variations_train_val_test.py"

VARIANTS=(
  base A1 A2 A3 A4 B1 B2
  C1 C2 C3 C4 C5 C6 C7
  D1 D2 D3 D4 E big
)

usage() {
  cat <<EOF
用法:
  $0 <variant> [传给 Python 脚本的额外参数...]
  $0 --foreground <variant> [额外参数...]
  $0 --list

可用变体:
  ${VARIANTS[*]}

示例:
  $0 base
  $0 C1 --train-steps 200 --max-train-samples 5000
  $0 A3 --batch-size 16 --accum-steps 16

说明:
  每次调用只运行一个变体。默认后台运行，日志保存到：
    results/<variant>/logs/<variant>_<时间>.log
  每个变体的结果、checkpoint、词表和日志都放在 results/<variant>/ 下。
  可用 LOG_DIR 环境变量修改日志目录。
  --foreground 表示前台运行，同时用 tee 写入同一日志。
EOF
}

FOREGROUND=0
if [[ "${1:-}" == "--foreground" ]]; then
  FOREGROUND=1
  shift
fi

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

case "$1" in
  -h|--help|help)
    usage
    exit 0
    ;;
  --list|list)
    printf '%s\n' "${VARIANTS[@]}"
    exit 0
    ;;
esac

VARIANT="$1"
shift

if [[ ! " ${VARIANTS[*]} " =~ " ${VARIANT} " ]]; then
  echo "错误: 未知变体 '$VARIANT'" >&2
  echo "运行 '$0 --list' 查看可用变体" >&2
  exit 1
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "错误: 未找到 Python: $PYTHON_BIN" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/results/$VARIANT/logs}"
mkdir -p "$LOG_DIR"
STAMP="$(date +"%Y%m%d_%H%M%S")"
LOG_FILE="$LOG_DIR/${VARIANT}_${STAMP}_$$.log"
PID_FILE="$LOG_DIR/${VARIANT}_${STAMP}_$$.pid"

{
  echo "variant=$VARIANT"
  echo "started=$(date -Is)"
  printf 'command:'
  printf ' %q' "$PYTHON_BIN" -u "$PY_SCRIPT" --variant "$VARIANT" "$@"
  echo
} > "$LOG_FILE"

CMD=("$PYTHON_BIN" -u "$PY_SCRIPT" --variant "$VARIANT" "$@")

if (( FOREGROUND )); then
  echo "日志文件: $LOG_FILE"
  "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
else
  nohup "${CMD[@]}" >> "$LOG_FILE" 2>&1 &
  PID=$!
  echo "$PID" > "$PID_FILE"
  echo "已在后台启动: $VARIANT"
  echo "PID: $PID"
  echo "PID 文件: $PID_FILE"
  echo "日志文件: $LOG_FILE"
  echo "查看日志: tail -f \"$LOG_FILE\""
fi
