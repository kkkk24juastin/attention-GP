#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-104.238.220.159}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_DIR="${REMOTE_DIR:-/opt}"
SSH_PORT="${SSH_PORT:-22}"
REMOTE_PASSWORD_FILE="${REMOTE_PASSWORD_FILE:-}"
DRY_RUN=0
PULL_ALL=0
SELECTED_CASE=""

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

usage() {
  cat <<'EOF'
用法:
  ./pull_results_from_opt.sh [选项] [案例编号或名称]

案例:
  1) 2d+10
  2) 2d+20
  3) 5d+20
  4) battery
  5) rocket
  6) wing_weight_10d

选项:
      --all             拉取全部常用案例的 refactored_results/
      --dry-run         只打印将执行的 scp 命令，不传输
      --password-file   从指定文件读取 SSH 密码，需本机已安装 sshpass
  -h, --help            显示帮助

环境变量:
  REMOTE_HOST           远端主机，默认 104.238.220.159
  REMOTE_USER           SSH 用户，默认 root
  REMOTE_DIR            远端项目根目录，默认 /opt
  SSH_PORT              SSH 端口，默认 22
  SSHPASS               SSH 密码，需本机已安装 sshpass
  REMOTE_PASSWORD_FILE  SSH 密码文件，需本机已安装 sshpass

示例:
  ./pull_results_from_opt.sh
  ./pull_results_from_opt.sh wing_weight_10d
  ./pull_results_from_opt.sh --all
  SSHPASS='你的密码' ./pull_results_from_opt.sh wing_weight_10d
  REMOTE_PASSWORD_FILE=/path/to/password.txt ./pull_results_from_opt.sh --all

说明:
  脚本只拉取远端案例目录下的 refactored_results/ 到本地同名案例目录。
  脚本不会保存密码。若未设置 SSHPASS 或 REMOTE_PASSWORD_FILE，将使用普通 scp 密码提示。
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
    printf '请选择要拉取结果的案例：\n' >&2
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

require_sshpass() {
  if ! command -v sshpass >/dev/null 2>&1; then
    printf '错误：已请求免输入密码，但本机未安装 sshpass。\n' >&2
    printf '建议改用 SSH key；或安装 sshpass 后再设置 SSHPASS/REMOTE_PASSWORD_FILE。\n' >&2
    exit 1
  fi
}

build_scp_prefix() {
  if [[ -n "${REMOTE_PASSWORD_FILE}" ]]; then
    require_sshpass
    printf '%s\n' sshpass -f "$REMOTE_PASSWORD_FILE"
  elif [[ -n "${SSHPASS:-}" ]]; then
    require_sshpass
    printf '%s\n' sshpass -e
  fi
}

pull_one_case() {
  local case_name="$1"
  local target_host="$2"
  local script_dir="$3"
  local local_case_dir="$script_dir/$case_name"
  local remote_results="${target_host}:${REMOTE_DIR%/}/${case_name}/refactored_results"
  local scp_prefix=()
  local scp_cmd=()

  if [[ ! -d "$local_case_dir" ]]; then
    printf '错误：本地案例目录不存在: %s\n' "$local_case_dir" >&2
    return 1
  fi

  mapfile -t scp_prefix < <(build_scp_prefix)
  scp_cmd=("${scp_prefix[@]}" scp -r -P "$SSH_PORT")
  scp_cmd+=("$remote_results" "$local_case_dir/")

  printf '\n案例: %s\n' "$case_name"
  printf '远端结果: %s\n' "$remote_results"
  printf '本地目录: %s\n' "$local_case_dir/refactored_results"
  printf '将执行: '
  quote_command "${scp_cmd[@]}"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi

  "${scp_cmd[@]}"
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --all)
      PULL_ALL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --password-file)
      if [[ $# -lt 2 ]]; then
        printf '错误：--password-file 需要一个文件路径。\n' >&2
        exit 1
      fi
      REMOTE_PASSWORD_FILE="$2"
      shift 2
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
      if [[ -n "$SELECTED_CASE" ]]; then
        printf '错误：只能指定一个案例，收到多余参数 "%s"。\n\n' "$1" >&2
        usage >&2
        exit 1
      fi
      SELECTED_CASE="$(resolve_case "$1")"
      shift
      ;;
  esac
done

if [[ "$PULL_ALL" -eq 1 && -n "$SELECTED_CASE" ]]; then
  printf '错误：--all 不能和单个案例同时使用。\n' >&2
  exit 1
fi

if [[ -n "$REMOTE_PASSWORD_FILE" && ! -r "$REMOTE_PASSWORD_FILE" ]]; then
  printf '错误：密码文件不可读: %s\n' "$REMOTE_PASSWORD_FILE" >&2
  exit 1
fi

if [[ "$PULL_ALL" -eq 0 && -z "$SELECTED_CASE" ]]; then
  SELECTED_CASE="$(prompt_case)"
fi

REMOTE_HOST="$(prompt_with_default "远端主机" "$REMOTE_HOST")"
REMOTE_USER="$(prompt_with_default "SSH 用户" "$REMOTE_USER")"
REMOTE_DIR="$(prompt_with_default "远端目录" "$REMOTE_DIR")"
SSH_PORT="$(prompt_with_default "SSH 端口" "$SSH_PORT")"

TARGET_HOST="$REMOTE_HOST"
if [[ -n "$REMOTE_USER" ]]; then
  TARGET_HOST="$REMOTE_USER@$REMOTE_HOST"
fi

printf '\n注意：拉取会覆盖本地同名结果文件。\n'
if [[ "$DRY_RUN" -eq 0 ]]; then
  printf '确认拉取远端结果？输入 yes 继续: '
  read -r answer
  if [[ "$answer" != "yes" ]]; then
    printf '已取消。\n'
    exit 0
  fi
fi

if [[ "$PULL_ALL" -eq 1 ]]; then
  for case_name in "${CASES[@]}"; do
    pull_one_case "$case_name" "$TARGET_HOST" "$SCRIPT_DIR"
  done
else
  pull_one_case "$SELECTED_CASE" "$TARGET_HOST" "$SCRIPT_DIR"
fi

printf '\n拉取完成。\n'
