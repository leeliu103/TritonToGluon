#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: dump-triton.sh --folder <path>

Run a Triton harness with dumping enabled and keep all artifacts.

Options:
  --folder     Folder containing *-harness.py (required)
  -h,--help    Show this message
EOF
}

err() {
  echo "Error: $1" >&2
  exit 1
}

FOLDER=""
PYTHON_BIN="${PYTHON:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --folder)
      [[ $# -ge 2 ]] || { usage; exit 1; }
      FOLDER="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

[[ -n "$FOLDER" ]] || { usage; exit 1; }
FOLDER="$(realpath "$FOLDER")"
[[ -d "$FOLDER" ]] || err "Folder not found: $FOLDER"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  err "Python executable not found: $PYTHON_BIN"
fi

mapfile -t HLIST < <(find "$FOLDER" -maxdepth 1 -type f -name "*-harness.py" | sort)
if [[ ${#HLIST[@]} -eq 0 ]]; then
  err "No *-harness.py found in $FOLDER"
elif [[ ${#HLIST[@]} -gt 1 ]]; then
  echo "Multiple harnesses found. Keep only one in the folder." >&2
  printf "  %s\n" "${HLIST[@]}" >&2
  exit 1
fi
HARNESS_PATH="${HLIST[0]}"

BASENAME="$(basename "$HARNESS_PATH" -triton-harness.py)"
BASENAME="${BASENAME%-harness}"

mapfile -t CLIST < <(find "$FOLDER" -maxdepth 1 -type f -name "${BASENAME}-config.json" | sort)
if [[ ${#CLIST[@]} -eq 0 ]]; then
  err "No config found: expected ${BASENAME}-config.json in $FOLDER"
elif [[ ${#CLIST[@]} -gt 1 ]]; then
  echo "Multiple configs found for base ${BASENAME}. Keep only one." >&2
  printf "  %s\n" "${CLIST[@]}" >&2
  exit 1
fi
CONFIG_PATH="$(realpath "${CLIST[0]}")"

mapfile -t SLIST < <(find "$FOLDER" -maxdepth 1 -type f -name "${BASENAME}-shape.json" | sort)
if [[ ${#SLIST[@]} -eq 0 ]]; then
  err "No shape found: expected ${BASENAME}-shape.json in $FOLDER"
elif [[ ${#SLIST[@]} -gt 1 ]]; then
  echo "Multiple shapes found for base ${BASENAME}. Keep only one." >&2
  printf "  %s\n" "${SLIST[@]}" >&2
  exit 1
fi
SHAPE_PATH="$(realpath "${SLIST[0]}")"

DUMP_DIR="$FOLDER/${BASENAME}-dump"
if [[ -d "$DUMP_DIR" ]]; then
  rm -rf "$DUMP_DIR"
fi
mkdir -p "$DUMP_DIR"

pushd "$FOLDER" >/dev/null
set +e
TRITON_DUMP_DIR="$DUMP_DIR" \
TRITON_ALWAYS_COMPILE=1 \
TRITON_KERNEL_DUMP=1 \
"$PYTHON_BIN" "$(basename "$HARNESS_PATH")" --config "$CONFIG_PATH" --shape "$SHAPE_PATH"
STATUS=$?
set -e
popd >/dev/null

if [[ $STATUS -ne 0 ]]; then
  err "Harness failed with exit code $STATUS"
fi

echo "Dumped Triton artifacts to: $DUMP_DIR"
