# Qwen3.8-Flash-Next W4A16 on 4× RTX 3090

[日本語](README_ja.md)

A port of [qwen38-flash-next-vllm-2x170hx](https://github.com/moriyasujapan/qwen38-flash-next-vllm-2x170hx)
to four RTX 3090s: TP4 + expert parallel, the same heretic-2 W4A16 checkpoint, the same
locally derived FP8 PLE offloaded to host RAM.

> **Nothing in this repository has been run on 3090 hardware.** It is a configuration
> study. Every number is either (a) measured on 2× CMP 170HX by the parent repo,
> (b) measured on 4× RTX 3090 by [alesha-pro](https://github.com/alesha-pro/qwen38-flash-next-4x3090),
> or (c) arithmetic on those, derived by `analysis/plan-memory.py`. Each claim below says
> which. Run `scripts/preflight.sh` on the target host first; it re-derives the fit table
> from the card sizes it actually finds.

---

## The one fact that decides the whole configuration

**At TP4 the KV cache per card is exactly the same size as at TP2.** It does not shrink.
From the pinned image, `vllm/models/qwen3_8_flash_next/nvidia/qsa.py:214-220`:

```python
self.total_num_kv_heads = int(config.num_key_value_heads)   # 2
if self.total_num_kv_heads >= tp_size:
    if self.total_num_kv_heads % tp_size:
        raise ValueError("QSA KV heads must be divisible by TP size")
elif tp_size % self.total_num_kv_heads:
    raise ValueError("TP size must be divisible by replicated QSA KV heads")
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

`num_key_value_heads` is 2, so `num_kv_heads` is `max(1, 2 // tp)` — **1 at TP2, 1 at TP4,
1 at TP8**. `k_proj`/`v_proj` are `(512, 2560)` = 2 heads × 256; past two ranks they are
replicated, not split. `common/qsa_cache.py:683` then *requires* exactly one KV head, which
is also why TP1 is rejected outright and TP must be even.

Consequence: going from 2 cards to 4 adds VRAM for **weights** and nothing for **context**.
Weights do shard cleanly, 37.43 → 19.48 GiB per card. But a 3090 is 24 GiB, so the slack
left over is ~3 GiB instead of the 170HX's ~25 GiB, and the per-token KV rate is unchanged
at 16,405 B (BF16) or 12,045 B (FP8) per card. One 262,144-token context needs 4.00 GiB of
BF16 KV per card. That does not fit, and no amount of extra cards makes it fit.

So the port is not "the same thing with `--tensor-parallel-size 4`". It is a choice between
**262K context** and **MTP speculative decoding**, which were the parent repo's two headline
features. You can have either. Not both, except at the very edge (profile D).

## What fits

`analysis/plan-memory.py`, 24 GiB cards, `--gpu-memory-utilization 0.98`, TP4:

| profile | context | MTP | KV dtype | KV avail | KV tokens | needed | verdict |
|---|---:|---|---|---:|---:|---:|---|
| **A** | 262,144 | off | fp8_e4m3 | 3.13 GiB | 279,288 | 2.94 GiB | fits, 1.07× |
| A' | 262,144 | off | BF16 | 3.13 GiB | 205,061 | 4.01 GiB | **no** |
| **B** | 131,072 | k=4 | fp8_e4m3 | 1.73 GiB | 154,041 | 1.47 GiB | fits, 1.18× |
| B+ | 196,608 | k=4 | fp8_e4m3 | 2.45 GiB | 218,759 | 2.21 GiB | fits, 1.11× — needs an unbuilt checkpoint |
| **C** | 98,304 | k=4 | BF16 | 1.73 GiB | 113,101 | 1.50 GiB | fits, 1.15× |
| C' | 65,536 | k=4 | BF16 | 1.45 GiB | 94,774 | 1.00 GiB | fits, 1.45× |
| D | 262,144 | k=4 | fp8_e4m3 | 2.82 GiB | 251,297 | 2.94 GiB | **no** at 0.98; fits at 0.99 |
| — | 262,144 | k=4 | BF16 | 12.6 GiB | 1,151,760 | 4.01 GiB | the 2x170hx default, for scale |

"contexts" of 1.07× means one full-length request and nothing else in flight. The parent
repo runs at 3.16×.

### Where the memory goes, per card

| | 2×170HX (TP2, measured) | 4×3090 (TP4, derived) |
|---|---:|---:|
| card | 63.4 GiB usable | 24.0 GiB |
| weights, MTP head included | 37.43 | **19.48** |
| non-torch (CUDA ctx, NCCL) | 1.27 | ~1.30 |
| prefill activations | 5.84 (8192 batched tok) | 0.73 (1024 batched tok) |
| CUDA-graph + decode buffers | 1.09 (seqs 8) | 0.14–0.28 (seqs 1–2) |
| **left for KV** | **17.60** | **1.73–3.13** |

The weight column is `analysis/weight-split.py`, which reads the safetensors headers and
applies the sharding policy taken out of the image. Its TP2 column reproduces the parent
repo's measured 38.7 GiB "weights + non-torch" to within 1.27 GiB, and that residual is
what the non-torch row uses. The activation slope is the parent repo's measured
1024 → 8192 batched-tokens step (KV had to drop 5.11 GiB). Back-solved against alesha-pro's
measured 4x3090 result the model predicts 279,288 FP8 KV tokens where they got **276,145** —
1.1% high.

```
analysis/weight-split.py --model-dir <checkpoint> --tp 2 --tp 4
analysis/plan-memory.py  --card-gib 24 --util 0.98
```

## Profiles

```bash
GPUS=0,1,2,3 PROFILE=a ./scripts/run.sh    # 262K, no MTP,  FP8 KV
GPUS=0,1,2,3 PROFILE=b ./scripts/run.sh    # 131K, MTP k=4, FP8 KV
GPUS=0,1,2,3 PROFILE=c ./scripts/run.sh    #  98K, MTP k=4, BF16 KV
```

**A — long context, no MTP.** The conservative choice, and the one with real 3090 evidence
behind it: it is the same operating point alesha-pro validated (262K configured, FP8 QSA KV,
`max-num-seqs 1`, `max-num-batched-tokens 1024`, util 0.98). Cost: the parent repo measured
**65.8 tok/s** with MTP off versus 101–170 with k=4. Dropping MTP is a 1.6× loss on prose
and 2.6× on code.

**B — keep MTP, halve the context.** The parent repo's whole decode advantage survives:
k=4 measured 101 tok/s ja prose / 170 tok/s code. 131,072 tokens is the ceiling the
arithmetic allows (154,041 available). This is the profile worth trying first if you came
from the 170HX setup and care about interactive latency.

**C — keep MTP, no FP8 KV at all.** BF16 KV, so no QSA overlay, no calibration file, no
FP8 KV accuracy question. 98,304 tokens. The simplest thing that could work; good first boot.

**B+ — 196,608 with MTP. Identified, not built.** The MTP draft head's MoE experts are
**BF16**, not quantized: `quantization_config.dynamic` is `{"-:.*mtp.*": {}}`, which excludes
everything matching `mtp` from the GPTQ scheme. Those two tensors —
`mtp.layers.48.mlp.experts.gate_up_proj` `(512,1280,2560)` and `.down_proj` `(512,2560,640)` —
are 4.69 GiB of the 4.86 GiB draft head, i.e. **1.17 of the 1.26 GiB per card that MTP costs
at TP4**. Storing them as FP8 E4M3 instead would free 0.59 GiB per card and move the MTP
ceiling from 131K to 196K.

What makes this unbuilt rather than easy: a plain dtype rewrite produces a corrupt
checkpoint. The module reads BF16 and would interpret FP8 bytes as garbage — the same trap
alesha-pro's builder works around by restoring 48 `shared_expert_gate` tensors to BF16. Doing
it properly means emitting a config the loader honours for those modules only, i.e. a mixed
`compressed-tensors` document with a GPTQ group and an FP8 group, and the image maps `gptq`
straight to `auto_gptq`. The image does carry `compressed-tensors`, `fp8` and `modelopt_mixed`
(`QUANTIZATION_METHODS`), so the pieces exist; whether `Qwen4ExpForConditionalGeneration`
loads a two-scheme document is untested. `PROFILE=b+` points at
`$MODELS_DIR/Qwen3.8-Flash-Next-heretic2-W4A16-mtpfp8` and fails cleanly until something
puts a checkpoint there.

**D — 262K with MTP, at the edge.** Needs all of: FP8 MTP experts (B+), FP8 KV,
`--max-num-batched-tokens 512`, `--max-num-seqs 1`, **and** util 0.99, which leaves ~240 MiB
on a 24 GiB card. The parent repo already found that vLLM's own "to fully utilize GPU memory"
suggestion OOMs on an 8.9K prefill with a 753 MiB margin. Expect this to OOM. Listed because
the arithmetic closes, not because it is advisable.

**Zero-VRAM speculation, worth trying under profile A.** `ngram`, `ngram_gpu` and `suffix`
are all in the image's `SpeculativeMethod` literals and need no draft weights, so they cost
no VRAM at all and leave profile A's 262K intact:

```bash
GPUS=0,1,2,3 PROFILE=a SPEC='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
  ./scripts/run.sh
```

The QSA ring sizing is identical for any k from 1 to 4 — `common/qsa_cache.py` computes
`capacity = compress_ratio * cdiv(compress_ratio + k, compress_ratio)` with
`compress_ratio = 4`, so k ∈ 1..4 all give 8 and only k = 0 gives 4. There is no KV reason
to lower k below 4.

## What changed from the 2×170HX recipe

| setting | 2×170HX | 4×3090 | why |
|---|---|---|---|
| `--tensor-parallel-size` | 2 | **4** | — |
| `--enable-expert-parallel` | required | **still required** | `moe_intermediate_size` 640 / 4 = 160, which the quant's group size 128 does not divide. EP keeps whole experts per rank (512 / 4 = 128) and 640 intact. |
| `--max-model-len` | 262144 | **98304–262144 by profile** | the KV fact above |
| `--speculative-config` | MTP k=4 | **off, or k=4 at reduced context** | 1.26 GiB/card of draft weights |
| `--kv-cache-dtype` | `auto` (BF16) | **`fp8_e4m3` for ≥131K** | BF16 262K needs 4.00 GiB/card, there are 3.13 |
| `--max-num-batched-tokens` | 8192 | **1024** | 0.73 GiB/card per 1024 tokens of prefill batch; seven more of those do not exist here |
| `--max-num-seqs` | 8 | **1–2** | ~0.14 GiB/card per slot, and the KV pool cannot hold 8 long requests anyway |
| `--gpu-memory-utilization` | 0.92 | **0.98** | the slack has to be pushed; run all four headless |
| `--kv-cache-memory` | 18897856102 explicit | **empty, let vLLM profile** | on the 170HX an explicit size bought capacity from real slack. Here the margin is 0.1–0.2 GiB and vLLM's profiling is the safe side. Set it by hand only if vLLM refuses `MAX_MODEL_LEN` for want of blocks. |
| `--pipeline-parallel-size` | refused | **still refused** | `overlays/gpu_worker.py:238` lists `PP=2` as unsupported under `VLLM_PLE_CPU_OFFLOAD`. Also refused: `nnodes≠1`, non-local DP, PCP, DCP, ubatching, weight transfer. TP is not restricted. |

Everything else — the image, the overlays, `VLLM_PLE_*`, `VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75`,
`--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}`, the `qwen3_xml` tool
parser, `SYS_PTRACE` — carries over verbatim.

## What transfers unchanged, and how that was checked

**The overlays are TP-generic.** `overlays/ple_offload/worker.py` validates
`num_workers == dp_size * tp_size` and sorts targets by `tp_rank`; nothing is hardcoded to 2.
The only parallelism `gpu_worker.py` rejects is the list above, and TP is not in it.

**Host RAM does not grow with TP.** This was the biggest risk: 48.4 GiB of FP8 PLE table
per rank would be 193 GiB at TP4. It does not happen —
`overlays/gpu_worker.py spawn_ple_offload()` returns early unless `self.rank == 0`, so there
is exactly **one node-local PLE worker** regardless of TP, and its docstring says so
("Own all discovered PLE tables and serve every local DP rank"). The per-TP-rank cost is one
pinned scratch buffer of `max_num_batched_tokens × ple_embed_dim`, which at 1024 × 2560 is
5 MiB. Budget the same ~61 GiB the TP2 container measured, plus two more worker processes:
call it 65–68 GiB, so 96 GiB of host RAM is comfortable and 128 GiB is the recommendation.

**The FP8 KV calibration transfers to TP4 as-is.** Both published calibrations record the
per-layer K/V absmax *maxed across TP ranks*, and since `num_kv_heads` is 1 at every TP ≥ 2
and `k_proj`/`v_proj` are not split past two ranks, that quantity is TP-invariant. Checked
numerically: the parent repo's TP2 file and alesha-pro's TP4 file agree to within 9% mean /
33% max per entry, and the parent's values are ≥ alesha's on **every one of the 12 shared
layers** (it was already built as the elementwise max of both). `scripts/merge-qsa-scales.py`
regenerates the merge and `results/fp8-kv-scales-tp4-mtp4.json` is the result — identical
per-layer to the parent repo's file, plus the provenance. It is also the **only** file with an
entry for the MTP drafter's attention layer (`mtp.layers.48`), which is exactly what profiles
B, B+ and D need.

**The image has sm_86 kernels.** 3090 is sm_86, the 170HX is sm_80, and the image is a custom
build. `torch.cuda.get_arch_list()` reports `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`,
`TORCH_CUDA_ARCH_LIST` is `7.5 8.0 8.6 8.9 9.0 10.0 12.0`, and
`cuobjdump --list-elf vllm/_C_stable_libtorch.abi3.so` lists an `sm_86.cubin`. `preflight.sh`
re-checks this.

**Head counts divide.** `num_attention_heads` 24 / 4 = 6. `linear_num_key_heads` 16 / 4 = 4,
`linear_num_value_heads` 48 / 4 = 12. `num_experts` 512 / 4 = 128. `vocab_size` 248320 /
4 = 62080. Vision tower `num_heads` 16 / 4 = 4, `intermediate_size` 4304 / 4 = 1076.
`shared_expert_intermediate_size` 640 / 4 = 160 is **not** a multiple of the group size 128,
but the shared expert is unquantized BF16 (only the 73,728 routed-expert `qweight` tensors
are GPTQ), so there is no group constraint on it. `indexer_n_heads` 4 and `indexer_kv_heads`
1 do not matter: `QSAIndexer` is documented as a "Replicated Q/K projection".

## Throughput: do not expect a prediction here

Two effects pull opposite ways and neither was measured on 3090s.

- **For 4×3090:** aggregate memory bandwidth. Decode is bandwidth-bound on the active
  weights, each card holds a quarter of them, and 4 × 936 GB/s is more aggregate GDDR6X than
  two HBM2e cards provide.
- **Against 4×3090:** interconnect. TP4 doubles the all-reduce count per layer versus TP2,
  and expert parallel across 4 ranks means a wider all-to-all for every MoE layer. The 170HX
  pair ran with **no P2P at all** (`can_device_access_peer` false both ways) on PCIe Gen2 x16
  and still served 100–170 tok/s, so this is survivable, but it is the term that gets worse.
  NVLink bridges pair up two 3090s at a time, which helps two of the six links.

The honest expectation is that profile A lands near the parent repo's measured no-MTP
**65.8 tok/s** and profile B near its **101 / 170 tok/s** prose / code figures, ± whatever
the interconnect does. Measure it:

```bash
until curl -sf localhost:18024/health; do sleep 10; done
python3 bench/bench.py --reps 3 --prefill 8192,32768 --conc 1,2,4
```

Note that profile B's 154,041 KV tokens still hold **four concurrent 33K requests** (132K),
so the parent repo's 392 tok/s-at-4-concurrent result is not obviously out of reach. What
4×3090 loses is long single contexts, not concurrency at moderate lengths.

## Quick start

```bash
git clone <this repo> && cd qwen38-flash-next-vllm-4x3090

GPUS=0,1,2,3 ./scripts/preflight.sh   # read-only; re-derives the fit table for your cards

./scripts/setup.sh              # pinned overlays + vLLM image (28.8 GB)
./scripts/download-model.sh     # base W4A16, 179.8 GB, resumable
./scripts/build-ple-fp8.sh      # BF16 PLE -> FP8 + manifest
./scripts/download-heretic2.sh  # heretic-2 W4A16, 74 GB, sha256-verified
./scripts/prepare-heretic2.sh   # GPTQ config, base tokenizer, MTP head graft

GPUS=0,1,2,3 PROFILE=c ./scripts/run.sh   # start with the simplest profile
docker logs -f flashnext-vllm
```

Endpoint `http://localhost:18024/v1`, model `flash-next-w4a16`. Startup took about 5 min 50 s
at TP2; expect more at TP4 (CUDA graph capture is per rank).

**Send `chat_template_kwargs: {"reasoning_effort": "medium"}`.** The chat template defaults
to `xhigh`, which the parent repo measured spending its entire 20,000-token budget thinking
without answering in 5 of 6 runs. This is unchanged by the port and it is the single setting
most likely to make the server look broken.

## Open questions

1. **The 0.1–0.2 GiB margin.** Every profile here is thinner than the parent repo's 753 MiB,
   which already OOMed one step up. If profile A will not boot at util 0.98, drop
   `--max-num-batched-tokens` to 512 before dropping the context.
2. **Is non-torch really ~1.30 GiB at TP4?** NCCL buffers grow with group size. If it is
   1.6 GiB, profile A loses ~27K KV tokens and still fits; profile B loses the same and still
   fits. Nothing breaks, but the margins shrink. `preflight.sh` cannot measure this; the
   engine log's own memory-profile line can, after one boot.
3. **FP8 KV accuracy at these scales, re-checked.** The parent repo found FP8 KV retrieval
   matched BF16 but decode 6–11% slower. Here it is not optional above 131K, and the slowdown
   lands on a configuration that already gave up MTP. The 12 shared layers' scales come from a
   TP4 calibration, so this is the best-supported part of the port — but it was calibrated on
   a different checkpoint composition.
4. **Does profile B+ exist?** See above. It is the only route to 196K+ with MTP.

## Credits

Everything the parent repo credits, plus:

- [alesha-pro/qwen38-flash-next-4x3090](https://github.com/alesha-pro/qwen38-flash-next-4x3090)
  for the PLE offload overlays this mounts, for the QSA FP8 KV path, and for the only
  measured 4×3090 data point this study calibrates against.
- [moriyasujapan/qwen38-flash-next-vllm-2x170hx](https://github.com/moriyasujapan/qwen38-flash-next-vllm-2x170hx)
  for the measurements every derivation here starts from.

## License

Apache-2.0 for the code in this repository. No weights are included; those are under the
Qwen Community License.
