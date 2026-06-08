#!/usr/bin/env bash
# Test the local frame-analysis prompt against the worker's Ollama, without touching the pipeline.
#
# Usage:
#   tools/test_frame_prompt.sh <frame.jpg> [n_runs] [prompt_file]
#
# Examples:
#   tools/test_frame_prompt.sh /mnt/nas/analyze-video/temp/20260528/09HBalcony_..._no_person_137.jpg
#   tools/test_frame_prompt.sh frame.jpg 5
#   tools/test_frame_prompt.sh frame.jpg 3 /tmp/prompt_test/prompt_v6.txt
#
# Env overrides (match analyze_frame.py defaults):
#   OLLAMA_URL (default http://192.168.1.25:11434)
#   OLLAMA_MODEL (default qwen3-vl:2b-instruct)
#   OLLAMA_TEMPERATURE (default 0.9), OLLAMA_TOP_K (default 20), OLLAMA_TOP_P (default 0.92),
#   OLLAMA_NUM_PREDICT (default 120), OLLAMA_REPEAT_PENALTY (default 1.3)
set -euo pipefail

FRAME="${1:?usage: test_frame_prompt.sh <frame.jpg> [n_runs] [prompt_file]}"
N_RUNS="${2:-1}"
PROMPT_FILE="${3:-$(dirname "$0")/../config/prompt_frame.txt}"

URL="${OLLAMA_URL:-http://192.168.1.25:11434}"
MODEL="${OLLAMA_MODEL:-qwen3-vl:2b-instruct}"
TEMP="${OLLAMA_TEMPERATURE:-0.8}"
TOP_K="${OLLAMA_TOP_K:-20}"
TOP_P="${OLLAMA_TOP_P:-0.92}"
NPREDICT="${OLLAMA_NUM_PREDICT:-120}"
REPEAT="${OLLAMA_REPEAT_PENALTY:-1.3}"
THINK="${OLLAMA_THINK:-}"   # set to "false" for gemma4 to skip its reasoning preamble

B64=$(mktemp)
trap 'rm -f "$B64"' EXIT
base64 -w0 "$FRAME" > "$B64"

echo "frame:  $(basename "$FRAME")"
echo "prompt: $PROMPT_FILE"
echo "model:  $MODEL  (temp=$TEMP top_k=$TOP_K top_p=$TOP_P num_predict=$NPREDICT repeat_penalty=$REPEAT think=${THINK:-<unset>})"
echo "---"

# Build the optional top-level `think` field (omitted unless OLLAMA_THINK is set)
THINK_JSON=""
if [ -n "$THINK" ]; then
  [ "$THINK" = "true" ] && THINK_JSON='"think":true,' || THINK_JSON='"think":false,'
fi

for ((i=1; i<=N_RUNS; i++)); do
  jq -n \
    --arg     model  "$MODEL" \
    --rawfile prompt "$PROMPT_FILE" \
    --rawfile image  "$B64" \
    --argjson temp   "$TEMP" \
    --argjson tk     "$TOP_K" \
    --argjson tp     "$TOP_P" \
    --argjson np     "$NPREDICT" \
    --argjson rp     "$REPEAT" \
    "{model:\$model, prompt:\$prompt, images:[\$image], stream:false, keep_alive:-1, ${THINK_JSON}
      options:{temperature:\$temp, top_k:\$tk, top_p:\$tp, num_predict:\$np, repeat_penalty:\$rp}}" \
  | curl -sN -X POST "$URL/api/generate" \
      -H 'Content-Type: application/json' \
      --data-binary @- \
  | jq -r 'if .error then "ERROR: \(.error)" else "[\(((.total_duration // 0)/1e9)|tostring|.[0:4])s] \(.response // "<empty> done_reason=\(.done_reason)")" end'
done
