#!/usr/bin/env bash
# Qwen3.8-Flash-Next W4A16 on four RTX 3090s, vLLM, TP4 + expert parallel, PLE
# offloaded to host RAM in FP8. Ported from the 2x CMP 170HX recipe; what had to
# change and why is in ../README.md. Short version: 4x24 GiB is 96 GiB, less than
# 2x64, and the KV cache does NOT get smaller per card at TP4 -- with
# num_key_value_heads=2 every TP >= 2 lands on exactly one KV head per rank
# (qsa.py:214-220), so TP4 only duplicates the same per-card cache across twice
# as many cards. After 19.5 GiB of weights there is ~3 GiB left per card, which
# is why the long-context profiles here give up either MTP or BF16 KV.
#
# PROFILE picks a tested-by-arithmetic operating point; every value stays
# individually overridable. analysis/plan-memory.py is where the numbers come
# from, and scripts/preflight.sh checks the host before you trust them.
#
#   PROFILE=a   262,144 ctx, MTP OFF,  FP8 KV   long context, ~66 tok/s decode
#   PROFILE=b   131,072 ctx, MTP k=4,  FP8 KV   keeps MTP: ~100 prose / ~170 code
#   PROFILE=b+  196,608 ctx, MTP k=4,  FP8 KV   needs prepare-mtp-fp8.sh first
#   PROFILE=c    98,304 ctx, MTP k=4,  BF16 KV  no FP8 KV calibration involved
#
# Expert parallel is not optional at TP4 either: moe_intermediate_size 640 split
# four ways is 160, which the quant's group size of 128 does not divide. With EP
# each rank owns whole experts (512/4 = 128) and the 640 stays intact.
set -euo pipefail
# Captured before env.sh defaults MODEL_DIR, so PROFILE=b+ can pick its own
# checkpoint without overriding an explicit MODEL_DIR from the caller.
_MODEL_DIR_SET=${MODEL_DIR:+1}
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

PROFILE=${PROFILE:-a}
case $PROFILE in
  a)  _CTX=262144; _SPEC='';                                              _KVD=fp8_e4m3; _SEQS=1 ;;
  b)  _CTX=131072; _SPEC='{"method":"mtp","num_speculative_tokens":4}';   _KVD=fp8_e4m3; _SEQS=2 ;;
  b+) _CTX=196608; _SPEC='{"method":"mtp","num_speculative_tokens":4}';   _KVD=fp8_e4m3; _SEQS=1
      [[ -z ${_MODEL_DIR_SET:-} ]] && MODEL_DIR=${MTP_FP8_DIR:-$MODELS_DIR/Qwen3.8-Flash-Next-heretic2-W4A16-mtpfp8} ;;
  c)  _CTX=98304;  _SPEC='{"method":"mtp","num_speculative_tokens":4}';   _KVD=auto;     _SEQS=2 ;;
  *)  echo "unknown PROFILE=$PROFILE (a|b|b+|c)" >&2; exit 2 ;;
esac

PLE_MANIFEST=${PLE_MANIFEST:-$PLE_MODEL_DIR/manifest.json}
CACHE=${CACHE:-$MODELS_DIR/.vllm-cache}
NAME=${NAME:-flashnext-vllm}
PORT=${PORT:-18024}
# The four 3090s, by index or UUID (`nvidia-smi -L`). TP caps every rank at the
# smallest card, so do not mix a 3090 with anything smaller.
GPUS=${GPUS:?set GPUS to the four RTX 3090 devices, e.g. GPUS=0,1,2,3 or their UUIDs}
TP=${TP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$_CTX}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-$_SEQS}
# 1024, not the 170HX's 8192: prefill activations cost ~0.73 GiB per card per
# 1024 batched tokens and there is no room for seven more of those here.
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-1024}
# 0.98, not 0.92: with only ~3 GiB of slack the fraction has to be pushed.
# Run the GPUs headless; a desktop on any of the four eats the whole margin.
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.98}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-$_KVD}
# Left empty on purpose. On the 170HX an explicit size was a win because there
# was slack to spend; here vLLM's own profiling is the safe side, and the
# measured per-card margin is 0.1-0.2 GiB. If vLLM refuses MAX_MODEL_LEN for
# want of KV blocks, size it by hand -- analysis/plan-memory.py prints the
# bytes, and 12045 B/token (FP8) or 16405 B/token (BF16) per card is the rate.
KV_CACHE_MEMORY=${KV_CACHE_MEMORY-}
COMPILATION_CONFIG=${COMPILATION_CONFIG:-'{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}'}
SPEC=${SPEC-$_SPEC}
# The merged calibration covers every full-attention layer plus the MTP
# drafter's, and is valid at TP4 -- see scripts/merge-qsa-scales.py.
QSA_SCALES=${QSA_SCALES:-$REPO/results/fp8-kv-scales-tp4-mtp4.json}
DIST=/usr/local/lib/python3.12/dist-packages
VLLM_PKG=$DIST/vllm

for f in "$MODEL_DIR/model.safetensors.index.json" \
         "$PLE_MODEL_DIR/model.safetensors.index.json" "$PLE_MANIFEST"; do
    [[ -f $f ]] || { echo "missing $f" >&2; exit 2; }
done
mkdir -p "$CACHE/vllm" "$CACHE/torchinductor"
docker rm -f "$NAME" 2>/dev/null || true

SPEC_ARGS=()
[[ -n $SPEC ]] && SPEC_ARGS=(--speculative-config "$SPEC")
KV_ARGS=()
[[ -n $KV_CACHE_MEMORY ]] && KV_ARGS=(--kv-cache-memory "$KV_CACHE_MEMORY")

QSA_MOUNTS=() QSA_ENV=()
if [[ $KV_CACHE_DTYPE == fp8* || -n ${QSA_CALIBRATE_OUT:-} ]]; then
    QSA_MOUNTS=(
        -v "$RECIPE/overlays/qwen38/qsa.py:$VLLM_PKG/models/qwen3_8_flash_next/nvidia/qsa.py:ro"
        -v "$RECIPE/overlays/qwen38/ops_qsa.py:$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ops/qsa.py:ro"
    )
fi
if [[ $KV_CACHE_DTYPE == fp8* ]]; then
    [[ -f ${QSA_SCALES:-} ]] || { echo "FP8 KV needs QSA_SCALES=<scales.json>" >&2; exit 2; }
    QSA_MOUNTS+=(-v "$(realpath "$QSA_SCALES"):/qsa-scales/scales.json:ro")
    QSA_ENV+=(-e QSA_FP8_SCALES_FILE=/qsa-scales/scales.json)
fi
if [[ -n ${QSA_CALIBRATE_OUT:-} ]]; then
    mkdir -p "$(dirname "$QSA_CALIBRATE_OUT")"
    QSA_MOUNTS+=(-v "$(realpath "$(dirname "$QSA_CALIBRATE_OUT")"):/qsa-calib")
    QSA_ENV+=(-e "QSA_FP8_CALIBRATE_OUT=/qsa-calib/$(basename "$QSA_CALIBRATE_OUT")")
fi
# shellcheck disable=SC2206
EXTRA=(${EXTRA_ARGS:-})

echo "profile $PROFILE: TP$TP ctx $MAX_MODEL_LEN seqs $MAX_NUM_SEQS" \
     "kv $KV_CACHE_DTYPE mtp ${SPEC:-off} model $(basename "$MODEL_DIR")" >&2

exec docker run -d --name "$NAME" --restart unless-stopped \
    --gpus "\"device=$GPUS\"" \
    --ipc=host \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
    -p "$PORT:$PORT" \
    -e VLLM_PLE_CPU_OFFLOAD=1 \
    -e VLLM_PLE_MODEL_PATH=/ple-model \
    -e VLLM_PLE_MANIFEST=/ple-manifest/manifest.json \
    -e VLLM_PLE_EMBEDDING_DTYPE=float8_e4m3fn \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75 \
    "${QSA_ENV[@]}" \
    -e HF_HOME=/hf-cache \
    -e TORCHINDUCTOR_CACHE_DIR=/compile-cache/torchinductor \
    -v "$MODEL_DIR:/model:ro" \
    -v "$PLE_MODEL_DIR:/ple-model:ro" \
    -v "$PLE_MANIFEST:/ple-manifest/manifest.json:ro" \
    -v "$CACHE:/hf-cache" \
    -v "$CACHE/vllm:/root/.cache/vllm" \
    -v "$CACHE/torchinductor:/compile-cache/torchinductor" \
    -v "$RECIPE/overlays/ple_layer.py:$VLLM_PKG/models/qwen3_8_flash_next/nvidia/ple_layer.py:ro" \
    -v "$RECIPE/overlays/ple_offload/worker.py:$VLLM_PKG/v1/ple_offload/worker.py:ro" \
    -v "$RECIPE/overlays/ple_offload/ple_external_source.py:$VLLM_PKG/v1/ple_offload/ple_external_source.py:ro" \
    -v "$RECIPE/overlays/gpu_worker.py:$VLLM_PKG/v1/worker/gpu_worker.py:ro" \
    -v "$RECIPE/overlays/multiproc_executor.py:$VLLM_PKG/v1/executor/multiproc_executor.py:ro" \
    -v "$RECIPE/overlays/qwen_gdn_linear_attn.py:$VLLM_PKG/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro" \
    -v "$RECIPE/overlays/shared_experts.py:$VLLM_PKG/model_executor/layers/fused_moe/runner/shared_experts.py:ro" \
    "${QSA_MOUNTS[@]}" \
    "$IMAGE" \
    /model \
    --served-model-name flash-next-w4a16 \
    --tensor-parallel-size "$TP" \
    --enable-expert-parallel \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-cache-dtype "$KV_CACHE_DTYPE" \
    --compilation-config "$COMPILATION_CONFIG" \
    "${SPEC_ARGS[@]}" \
    "${KV_ARGS[@]}" \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    --no-enable-flashinfer-autotune \
    "${EXTRA[@]}" \
    --host 0.0.0.0 \
    --port "$PORT"
