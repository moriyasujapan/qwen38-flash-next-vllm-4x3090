#!/usr/bin/env python3
"""What context length fits on N cards of a given size, for this checkpoint.

Everything here is anchored on measurements, not on vLLM's profiler:

KV bytes per token PER CARD, 16405 B (BF16)
    Two independent 2x170hx data points agree to 4 bytes:
      24383208960 / 1486412 = 16403    18897856102 / 1151760 = 16407
    This number is TENSOR-PARALLEL INVARIANT for this model. qsa.py:214-220 sets
      num_kv_heads = max(1, num_key_value_heads // tp_size)
    and num_key_value_heads is 2, so every TP >= 2 lands on exactly one KV head
    per rank. TP4 buys no KV capacity per card over TP2 -- it only duplicates
    the cache across twice as many cards. (qsa_cache.py:683 also *requires*
    exactly one KV head, so TP1 is rejected outright.)

KV bytes per token per card, 12045 B (fp8_e4m3)
    From the measured +36.2% capacity (1058505 -> 1441792 tokens). The Gated
    Delta Net state stays BF16, which is why it is not a clean 2x.

Overhead, GiB per card
    1.30  non-torch: CUDA context, NCCL buffers. 2x170hx measured 38.7 GiB of
          "weights + non-torch" where the tensor model says 37.43.
    0.73  per 1024 max-num-batched-tokens, prefill activations. From the
          measured step 1024 -> 8192 batched tokens needing 5.11 GiB more room
          (KV 22.709 -> 17.601 GiB per card).
    0.14  per max-num-seqs slot, FULL_DECODE_ONLY graph pools and decode
          buffers. Back-solved from the same peak (64783 MiB at seqs=8).

    Cross-check against the only real 4x3090 datapoint, alesha-pro's recipe
    (TP4, no MTP, fp8 KV, bt=1024, seqs=1, util 0.98): this model predicts
    279.8K KV tokens, they measured 276,145. 1.3% high.

Not measured on 3090 hardware. Treat every number as a starting point with a
+-0.3 GiB uncertainty band, and confirm with scripts/preflight.sh on the host.
"""
import argparse

GiB = 2 ** 30
KV_BYTES = {"auto": 16405, "bf16": 16405, "fp8_e4m3": 12045, "fp8": 12045}

# analysis/weight-split.py, TP4 column
WEIGHTS_TP4 = {"mtp": 19.482, "mtp-fp8-experts": 18.896, "no-mtp": 18.217}

NONTORCH = 1.30
ACT_PER_1K = 0.73
GRAPH_PER_SEQ = 0.14


def overhead(batched_tokens, max_num_seqs):
    return (NONTORCH
            + ACT_PER_1K * batched_tokens / 1024
            + GRAPH_PER_SEQ * max_num_seqs)


def kv_available(card_gib, util, variant, batched_tokens, max_num_seqs):
    return card_gib * util - WEIGHTS_TP4[variant] - overhead(batched_tokens, max_num_seqs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card-gib", type=float, default=24.0, help="per-GPU VRAM")
    ap.add_argument("--util", type=float, default=0.98)
    args = ap.parse_args()

    rows = [
        ("A   262K, no MTP",            "no-mtp",          "fp8_e4m3", 1024, 1, 262144),
        ("A'  262K, no MTP, BF16 KV",   "no-mtp",          "auto",     1024, 1, 262144),
        ("B   131K, MTP k=4",           "mtp",             "fp8_e4m3", 1024, 2, 131072),
        ("B+  196K, MTP + FP8 experts", "mtp-fp8-experts", "fp8_e4m3", 1024, 1, 196608),
        ("C   98K,  MTP, BF16 KV",      "mtp",             "auto",     1024, 2,  98304),
        ("C'  64K,  MTP, BF16 KV",      "mtp",             "auto",     1024, 4,  65536),
        ("D   262K, MTP, everything",   "mtp-fp8-experts", "fp8_e4m3",  512, 1, 262144),
    ]

    print(f"{args.card_gib:.0f} GiB cards, --gpu-memory-utilization {args.util}, TP4\n")
    hdr = (f"{'profile':<30}{'KV avail':>10}{'KV tokens':>11}"
           f"{'need':>10}{'contexts':>10}  verdict")
    print(hdr); print("-" * len(hdr))
    for label, variant, kvd, bt, seqs, ctx in rows:
        avail = kv_available(args.card_gib, args.util, variant, bt, seqs)
        per = KV_BYTES[kvd]
        tokens = int(avail * GiB / per) if avail > 0 else 0
        need = ctx * per / GiB
        fits = tokens >= ctx
        print(f"{label:<30}{avail:>9.3f}G{tokens:>11,}{need:>9.2f}G"
              f"{(tokens/ctx if ctx else 0):>9.2f}x  "
              f"{'OK' if fits else 'DOES NOT FIT'}")

    # The calibration this model is fitted to: 2x170hx, TP2, MTP k=4, BF16 KV,
    # bt=8192, seqs=8, observed peak 64783 MiB of 65536 on a 63.4 GiB-usable card.
    cal = 63.26 - 37.429 - overhead(8192, 8)
    print(f"\ncalibration  2x170hx TP2 peak: model {cal:.3f}G KV "
          f"({int(cal*GiB/16405):,} tokens)  measured 17.601G (1,151,760) "
          f"-> {100*(cal/17.601-1):+.1f}%")

    print("\nsensitivity, profile A/B headroom per GiB of card:")
    for u in (0.96, 0.97, 0.98, 0.99):
        a = kv_available(args.card_gib, u, "no-mtp", 1024, 1)
        b = kv_available(args.card_gib, u, "mtp", 1024, 2)
        print(f"  util {u}:  no-MTP {max(0,int(a*GiB/12045)):>8,} fp8 tokens"
              f"   MTP {max(0,int(b*GiB/12045)):>8,} fp8 tokens")


if __name__ == "__main__":
    main()
