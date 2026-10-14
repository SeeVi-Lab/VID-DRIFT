#!/usr/bin/env bash
# Run every video-LLM eval on GPUs 0 and 2 (--limit rows each), then compare models.
#
# Usage:
#   bash run_all.sh                         # restored (DOVE) videos, defaults below
#   bash run_all.sh --upscaler none         # extra args are forwarded to every model script
#   MAIN_ENV=vlm MOLMO_ENV=vmolmo OV2_ENV=vlm_tf5 CAMBRIAN_ENV=cambrians bash run_all.sh
#   RUN_MODELS="qwen3vl molmo2" bash run_all.sh      # subset
#
# Notes:
#   - prompts/scoring follow the VSTAT lmms-eval task (--protocol vstat is the default)
#   - Gemini runs only if GEMINI_API_KEY or GOOGLE_API_KEY is set (paid API, no GPU)
#   - --limit counts evaluated rows, and rows run item by item (all conditions of an
#     item back to back), so 300 rows ~= 300 / n_conditions fully paired items.
#   - Rerunning resumes: already-scored rows are skipped and up to LIMIT new rows run.
#   - HF models use device_map="auto" (split across both visible GPUs); Cambrian-S uses
#     its repo's device_map="cuda", i.e. only physical GPU 0.
set -uo pipefail

cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0,1
LIMIT="${LIMIT:-0}"
CSV="${CSV:-../DS_csv/downsample.csv}"
OUTDIR="${OUTDIR:-EDEN_mvbench_limit${LIMIT}}"
MAIN_ENV="${MAIN_ENV:-}"            # Qwen3-VL, LLaVA-OV, InternVL3, Gemini, compare
MOLMO_ENV="${MOLMO_ENV:-$MAIN_ENV}" # transformers == 4.57.1 (Molmo2 model-card pin): Molmo2
OV2_ENV="${OV2_ENV:-$MAIN_ENV}"     # transformers >= 5.7: LLaVA-OneVision-2
CAMBRIAN_ENV="${CAMBRIAN_ENV:-$MAIN_ENV}"  # cambrian-s repo installed: Cambrian-S
# which models to run (space-separated); gemini is skipped unless an API key is set
RUN_MODELS="${RUN_MODELS:-qwen3vl internvl3 molmo2 llava_ov2}"
EXTRA_ARGS=("$@")

LOGDIR="$OUTDIR/logs"
mkdir -p "$LOGDIR"

activate() {
  if [[ -n "$1" ]]; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$1"
  fi
}

# name | script | env   (frame budgets / resolutions are each script's VSTAT defaults)
MODELS=(
  "qwen3vl|qwen3vl_generation.py|$MAIN_ENV"
  "llava_ov|llava_ov_generation.py|$MAIN_ENV"
  "internvl3|internvl3_generation.py|$MAIN_ENV"
  "molmo2|molmo2_generation.py|$MOLMO_ENV"
  "llava_ov2|llava_ov2_generation.py|$OV2_ENV"
  "cambrian_s|cambrian_s_generation.py|$CAMBRIAN_ENV"
  "gemini|gemini_generation.py|$MAIN_ENV"
)

RESULTS=()
FAILED=()
for spec in "${MODELS[@]}"; do
  IFS="|" read -r name script env <<< "$spec"
  [[ " $RUN_MODELS " == *" $name "* ]] || continue
  if [[ $name == gemini && -z "${GEMINI_API_KEY:-}${GOOGLE_API_KEY:-}" ]]; then
    echo "=== gemini skipped (set GEMINI_API_KEY or GOOGLE_API_KEY to run it) ==="
    continue
  fi
  out="$OUTDIR/${name}_results.csv"
  echo "=== $name  [$(date '+%H:%M:%S')]  GPUs $CUDA_VISIBLE_DEVICES  -> $out ==="
  (
    activate "$env"
    python "$script" --csv "$CSV" --out "$out" --limit "$LIMIT" --show 3 \
      ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
  ) 2>&1 | tee "$LOGDIR/$name.log"
  rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "!!! $name exited with code $rc (see $LOGDIR/$name.log); continuing"
    FAILED+=("$name")
  fi
  [[ -s "$out" ]] && RESULTS+=("$out")
done

echo
echo "=== compare  [$(date '+%H:%M:%S')] ==="
if [[ ${#RESULTS[@]} -eq 0 ]]; then
  echo "No result files produced; nothing to compare."
  exit 1
fi
( activate "$MAIN_ENV"; python compare_eval.py "${RESULTS[@]}" ) 2>&1 | tee "$OUTDIR/comparison.txt"

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo
  echo "Failed models: ${FAILED[*]}"
fi