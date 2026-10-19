#!/usr/bin/env python3
"""Build the FP8 QSA KV scales file this repo serves with, from the two published
calibrations, taking per layer and per tensor the LARGER absmax.

Why the TP2 calibration is valid at TP4: both files record "max across TP ranks"
of the per-layer K/V absmax, and qsa.py:214-220 gives num_kv_heads=1 at every
TP >= 2 because num_key_value_heads is 2. k_proj/v_proj are not split past two
ranks, so each rank sees one of the same two KV heads either way -- a per-layer
max over ranks is the same quantity at TP2 and TP4. The two calibrations agree
to within 0-28% per entry, and taking the max is the safe direction: clipping an
outlier costs more accuracy than losing a little mantissa resolution.

Only the 2x170hx file has an entry for the MTP drafter's attention layer
(mtp.layers.48), which is exactly what the MTP-enabled 3090 profiles need.

    ./merge-qsa-scales.py a.json b.json -o results/fp8-kv-scales-tp4-mtp4.json
"""
import argparse, json, sys

FP8_E4M3_MAX = 448.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", required=True)
    args = ap.parse_args()

    merged, sources, agree = {}, [], []
    for path in args.inputs:
        doc = json.load(open(path))
        sources.append({"file": path, "provenance": doc.get("provenance")})
        for layer, entry in doc["per_layer"].items():
            cur = merged.setdefault(layer, {})
            for side in ("k", "v"):
                amax = float(entry[f"{side}_absmax"])
                prev = cur.get(f"{side}_absmax")
                if prev is not None:
                    agree.append(abs(amax - prev) / max(amax, prev))
                if prev is None or amax > prev:
                    cur[f"{side}_absmax"] = amax
                    cur[f"{side}_scale"] = amax / FP8_E4M3_MAX

    out = {
        "per_layer": dict(sorted(merged.items())),
        "default": {
            "k_scale": max(e["k_absmax"] for e in merged.values()) / FP8_E4M3_MAX,
            "v_scale": max(e["v_absmax"] for e in merged.values()) / FP8_E4M3_MAX,
        },
        "provenance": (
            "per-entry maximum of the inputs below; scale = absmax/448. Valid at "
            "any TP >= 2 for this checkpoint because num_key_value_heads=2 "
            "resolves to num_kv_heads=1 on every rank (qsa.py:214-220), so the "
            "per-layer max-across-ranks absmax is TP-invariant. Merged by "
            "scripts/merge-qsa-scales.py."
        ),
        "sources": sources,
    }
    json.dump(out, open(args.out, "w"), indent=1)
    n = len(merged)
    print(f"{n} layers -> {args.out}")
    if agree:
        print(f"inputs overlap on {len(agree)//2} layers; per-entry disagreement "
              f"max {100*max(agree):.1f}%, mean {100*sum(agree)/len(agree):.1f}%")


if __name__ == "__main__":
    main()
