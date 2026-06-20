#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-104.238.220.159}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/opt}"
SSH_PORT="${SSH_PORT:-22}"
DRY_RUN=0

CASES=(
  "2d+10"
  "2d+20"
  "5d+20"
  "battery"
  "rocket"
  "wing_weight_10d"
)

LEGACY_CASES=(
  "5d+10"
)

COMMON_EXCLUDES=(
  "__pycache__/"
  "*.pyc"
  "*.pyo"
  "*.pyd"
)

ROCKET_EXCLUDES=(
  "generated_data/initial_openrocket/"
)

usage() {
  cat <<'EOF'
用法:
  ./upload_case_to_opt.sh [选项]

案例:
  1) 2d+10
  2) 2d+20
  3) 5d+20
  4) battery
  5) rocket
  6) wing_weight_10d

选项:
      --dry-run         只打印将执行的 scp 命令，不传输
  -h, --help            显示帮助

示例:
  ./upload_case_to_opt.sh
  ./upload_case_to_opt.sh --dry-run

说明:
  脚本使用 rsync 的密码登录流程，不读取、不保存密码。
  SSH 用户默认是 root；如果要用其他账号，请手动填写可登录账号。
  会自动跳过编译产物和 rocket/generated_data/initial_openrocket/。
  运行后请根据终端提示输入远端用户密码。
EOF
}

list_cases() {
  local i
  for i in "${!CASES[@]}"; do
    printf '%d) %s\n' "$((i + 1))" "${CASES[$i]}"
  done
}

quote_command() {
  local arg
  printf '%q' "$1"
  shift
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}

resolve_case() {
  local input="$1"
  local i

  if [[ "$input" =~ ^[1-6]$ ]]; then
    printf '%s\n' "${CASES[$((input - 1))]}"
    return 0
  fi

  for i in "${!CASES[@]}"; do
    if [[ "$input" == "${CASES[$i]}" ]]; then
      printf '%s\n' "${CASES[$i]}"
      return 0
    fi
  done

  for i in "${!LEGACY_CASES[@]}"; do
    if [[ "$input" == "${LEGACY_CASES[$i]}" ]]; then
      printf '%s\n' "${LEGACY_CASES[$i]}"
      return 0
    fi
  done

  printf '错误：未知案例 "%s"。\n\n' "$input" >&2
  list_cases >&2
  printf '\n兼容旧目录: %s\n' "${LEGACY_CASES[*]}" >&2
  return 1
}

prompt_case() {
  local choice

  while true; do
    printf '请选择要上传的案例：\n' >&2
    list_cases >&2
    printf '输入编号或名称: ' >&2
    read -r choice

    if resolve_case "$choice"; then
      return 0
    fi
    printf '\n' >&2
  done
}

prompt_with_default() {
  local prompt="$1"
  local default="$2"
  local value

  printf '%s [%s]: ' "$prompt" "$default" >&2
  read -r value
  printf '%s\n' "${value:-$default}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      printf '错误：未知选项 "%s"。\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
    *)
      printf '错误：本脚本为交互式选择，不接受位置参数 "%s"。\n\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

SELECTED_CASE="$(prompt_case)"

SOURCE_PATH="$SCRIPT_DIR/$SELECTED_CASE"
if [[ ! -d "$SOURCE_PATH" ]]; then
  printf '错误：本地案例目录不存在: %s\n' "$SOURCE_PATH" >&2
  exit 1
fi

REMOTE_HOST="$(prompt_with_default "远端主机" "$REMOTE_HOST")"
REMOTE_USER="$(prompt_with_default "SSH 用户" "$REMOTE_USER")"
REMOTE_DIR="$(prompt_with_default "远端目录" "$REMOTE_DIR")"
SSH_PORT="$(prompt_with_default "SSH 端口" "$SSH_PORT")"

TARGET_HOST="$REMOTE_HOST"
if [[ -n "$REMOTE_USER" ]]; then
  TARGET_HOST="$REMOTE_USER@$REMOTE_HOST"
fi
TARGET="${TARGET_HOST}:${REMOTE_DIR%/}/"

RSYNC_SSH=(
  "ssh"
  "-p" "$SSH_PORT"
  "-o" "BatchMode=no"
  "-o" "PasswordAuthentication=yes"
  "-o" "KbdInteractiveAuthentication=yes"
  "-o" "PreferredAuthentications=publickey,password,keyboard-interactive"
  "-o" "NumberOfPasswordPrompts=3"
)
RSYNC_CMD=(rsync -a -e "${RSYNC_SSH[*]}")

EXCLUDES=("${COMMON_EXCLUDES[@]}")
if [[ "$SELECTED_CASE" == "rocket" ]]; then
  EXCLUDES+=("${ROCKET_EXCLUDES[@]}")
fi

for pattern in "${EXCLUDES[@]}"; do
  RSYNC_CMD+=(--exclude="$pattern")
done

RSYNC_CMD+=("$SOURCE_PATH" "$TARGET")

printf '本地案例: %s\n' "$SOURCE_PATH"
printf '远端目录: %s\n' "$TARGET"
printf '将执行: '
quote_command "${RSYNC_CMD[@]}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

printf '确认上传并写入远端目录？输入 yes 继续: '
read -r answer
if [[ "$answer" != "yes" ]]; then
  printf '已取消。\n'
  exit 0
fi

printf '即将启动 rsync。出现 password 提示时，请输入远端用户密码。\n'
exec "${RSYNC_CMD[@]}"
