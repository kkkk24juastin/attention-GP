#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$SCRIPT_DIR/build"

CLEAN=1
FULL_CLEAN=0

usage() {
  printf '%s\n' \
    "Usage: ./build_latex.sh [--no-clean] [--full-clean] [--outdir DIR]" \
    "" \
    "Builds interactcadsample.tex and interactcadsample_cn.tex." \
    "PDFs are written to ./build by default." \
    "" \
    "Options:" \
    "  --no-clean    Keep auxiliary files after build." \
    "  --full-clean  Remove PDFs and auxiliary files, then exit." \
    "  --outdir DIR  Write generated files to DIR."
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-clean)
      CLEAN=0
      ;;
    --full-clean)
      FULL_CLEAN=1
      ;;
    --outdir)
      if [ "$#" -lt 2 ]; then
        printf 'Error: --outdir requires a directory.\n' >&2
        exit 2
      fi
      OUT_DIR="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Error: unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

if ! command -v latexmk >/dev/null 2>&1; then
  printf 'Error: latexmk is required but was not found in PATH.\n' >&2
  exit 127
fi

mkdir -p "$OUT_DIR"

cd "$SCRIPT_DIR"

if [ "$FULL_CLEAN" -eq 1 ]; then
  latexmk -C -outdir="$OUT_DIR" interactcadsample.tex interactcadsample_cn.tex
  exit 0
fi

build_pdf() {
  local engine="$1"
  local tex_file="$2"

  printf 'Building %s with %s...\n' "$tex_file" "$engine"
  latexmk "$engine" -interaction=nonstopmode -halt-on-error -outdir="$OUT_DIR" "$tex_file"
}

build_pdf -pdf interactcadsample.tex
build_pdf -xelatex interactcadsample_cn.tex

if [ "$CLEAN" -eq 1 ]; then
  latexmk -c -outdir="$OUT_DIR" interactcadsample.tex interactcadsample_cn.tex
fi

printf 'Done. PDFs:\n'
printf '  %s\n' "$OUT_DIR/interactcadsample.pdf"
printf '  %s\n' "$OUT_DIR/interactcadsample_cn.pdf"
