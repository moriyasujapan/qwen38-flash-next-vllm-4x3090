#!/usr/bin/env python3
"""Per-rank weight footprint of the served checkpoint at a given tensor-parallel size.

Reads the safetensors headers directly (no torch, no GPU) and applies the sharding
policy the vLLM overlays actually use, which was read out of the image:

  routed experts (`.mlp.experts.`)   expert-parallel, 512 experts split by TP
  q_proj / o_proj                    24 heads, split by TP
  k_proj / v_proj                    num_key_value_heads=2, so split by min(TP,2)
                                     and REPLICATED beyond that (qsa.py:214-220)
  linear_attn in_proj_*/out_proj     16 key / 48 value heads, split by TP
  shared_expert, vision, lm_head,
  embed_tokens                       split by TP
  indexer index_qk_proj              replicated ("Replicated Q/K projection",
                                     indexer_qsa.py QSAIndexer docstring)
  hyperconnection, norms, gates,
  convs, ple key/value proj          replicated

Calibration: the TP2 column reproduces the 2x170hx measurement of 38.7 GiB
"weights + non-torch" per card to within the non-torch allowance (~1.3 GiB).

    ./weight-split.py --model-dir /path/to/checkpoint --tp 2 --tp 4
"""
import argparse, collections, glob, json, os, re, struct, sys

ELEM = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "F8_E4M3": 1, "F8_E5M2": 1,
        "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
GiB = 2 ** 30

NUM_KV_HEADS = 2        # text_config.num_key_value_heads
NUM_Q_HEADS = 24        # text_config.num_attention_heads

TP_SPLIT = (
    r"self_attn\.(q_proj|o_proj)\.weight$",
    r"linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight$",
    r"mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight$",
    r"^(lm_head\.weight|model\.language_model\.embed_tokens\.weight)$",
    r"visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc[12])\.weight$",
    r"visual\.merger\.linear_fc[12]\.weight$",
)


def read_headers(model_dir):
    out = {}
    shards = sorted(glob.glob(os.path.join(model_dir, "model-*.safetensors")))
    if not shards:
        sys.exit(f"no model-*.safetensors under {model_dir}")
    for path in shards:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, spec in hdr.items():
            if name == "__metadata__":
                continue
            count = 1
            for dim in spec["shape"]:
                count *= dim
            out[name] = (count * ELEM[spec["dtype"]], spec["dtype"])
    return out


def policy(name):
    if ".mlp.experts." in name:
        return "EP"
    if re.search(r"self_attn\.(k_proj|v_proj)\.weight$", name):
        return "KV"
    if any(re.search(p, name) for p in TP_SPLIT):
        return "TP"
    return "REP"


def per_rank(tensors, tp, drop_mtp=False, fp8_mtp_experts=False):
    acc = collections.Counter()
    for name, (nbytes, dtype) in tensors.items():
        is_mtp = name.startswith("mtp.")
        if drop_mtp and is_mtp:
            continue
        if fp8_mtp_experts and is_mtp and ".mlp.experts." in name and dtype == "BF16":
            nbytes //= 2
        kind = policy(name)
        if kind in ("EP", "TP"):
            acc[kind] += nbytes / tp
        elif kind == "KV":
            acc[kind] += nbytes / min(tp, NUM_KV_HEADS)
        else:
            acc[kind] += nbytes
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tp", type=int, action="append", default=None)
    args = ap.parse_args()
    tps = args.tp or [2, 4]

    t = read_headers(args.model_dir)
    disk = sum(b for b, _ in t.values())
    mtp = sum(b for n, (b, _) in t.items() if n.startswith("mtp."))
    print(f"{len(t)} tensors, {disk/GiB:.3f} GiB on disk "
          f"(MTP draft head {mtp/GiB:.3f} GiB)\n")

    if NUM_Q_HEADS % max(tps):
        print(f"WARNING: {NUM_Q_HEADS} attention heads do not divide TP={max(tps)}\n")

    hdr = f"{'variant':<34}" + "".join(f"{'TP'+str(x):>12}" for x in tps)
    print(hdr)
    print("-" * len(hdr))
    for label, kw in (("full (MTP head, BF16)", {}),
                      ("MTP experts FP8", dict(fp8_mtp_experts=True)),
                      ("no MTP head", dict(drop_mtp=True))):
        row = f"{label:<34}"
        for tp in tps:
            row += f"{sum(per_rank(t, tp, **kw).values())/GiB:>11.3f}G"
        print(row)

    print()
    for tp in tps:
        a = per_rank(t, tp)
        print(f"TP{tp} breakdown, GiB per rank: " + "  ".join(
            f"{k}={a[k]/GiB:.3f}" for k in ("EP", "TP", "KV", "REP")))
        print(f"      replicated-and-duplicated mass: "
              f"{(a['REP']*tp + a['KV']*tp - a['REP'] - a['KV'])/GiB:.3f} GiB "
              f"wasted across the group vs one copy")


if __name__ == "__main__":
    main()
