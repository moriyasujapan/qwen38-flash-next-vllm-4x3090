#!/usr/bin/env bash
# Host checks for the 4x3090 profiles. Everything in README.md is arithmetic on
# measurements taken on other hardware; this is the part that has to be true on
# YOUR machine before those numbers mean anything. Read-only, starts nothing.
#
#   GPUS=0,1,2,3 ./scripts/preflight.sh
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"

fail=0 warn=0
ok()   { printf '  \033[32mok\033[0m    %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=$((fail+1)); }
note() { printf '  \033[33mwarn\033[0m  %s\n' "$*"; warn=$((warn+1)); }
hdr()  { printf '\n%s\n' "$*"; }

GPUS=${GPUS:-0,1,2,3}
IFS=',' read -r -a GPU_ARR <<< "$GPUS"

hdr "GPUs ($GPUS)"
if ! command -v nvidia-smi >/dev/null; then
    bad "nvidia-smi not found"
else
    [[ ${#GPU_ARR[@]} -eq 4 ]] || bad "TP4 needs 4 devices, GPUS lists ${#GPU_ARR[@]}"
    q=$(nvidia-smi -i "$GPUS" --format=csv,noheader \
        --query-gpu=index,name,memory.total,memory.used,compute_cap,pcie.link.gen.current,pcie.link.width.current,display_active 2>/dev/null)
    if [[ -z $q ]]; then
        bad "nvidia-smi could not query devices $GPUS"
    else
        printf '%s\n' "$q" | sed 's/^/        /'
        mapfile -t caps  < <(printf '%s\n' "$q" | awk -F', ' '{print $5}')
        mapfile -t tots  < <(printf '%s\n' "$q" | awk -F', ' '{gsub(/ MiB/,"");print $3}')
        mapfile -t useds < <(printf '%s\n' "$q" | awk -F', ' '{gsub(/ MiB/,"");print $4}')
        mapfile -t disp  < <(printf '%s\n' "$q" | awk -F', ' '{print $8}')
        smallest=999999
        for i in "${!tots[@]}"; do
            (( tots[i] < smallest )) && smallest=${tots[i]}
            # TP caps every rank at the smallest card.
            (( tots[i] < 24000 )) && bad "device ${GPU_ARR[i]} has only ${tots[i]} MiB; TP4 sizes every rank to the smallest card"
            # qsa.py needs sm_80+; Marlin W4A16 needs sm_80+. 3090 is 8.6.
            awk -v c="${caps[i]}" 'BEGIN{exit !(c+0 >= 8.0)}' \
                || bad "device ${GPU_ARR[i]} compute_cap ${caps[i]} < 8.0"
            # Every MiB in use is a MiB off a 0.1-0.2 GiB margin.
            (( useds[i] > 300 )) && note "device ${GPU_ARR[i]} already has ${useds[i]} MiB in use"
            [[ ${disp[i]} == Enabled ]] && bad "device ${GPU_ARR[i]} has an active display; run these four headless"
        done
        (( smallest < 999999 )) && ok "smallest card ${smallest} MiB -> plan with --card-gib $(awk -v m=$smallest 'BEGIN{printf "%.2f", m/1024}')"
        uniq_n=$(printf '%s\n' "${tots[@]}" | sort -u | wc -l)
        (( uniq_n > 1 )) && note "cards are not all the same size; TP4 wastes the difference"
    fi
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "${GPU_ARR[0]}" 2>/dev/null | head -1)
    # The image is CUDA 13.0; its forward-compat floor is the r580 branch.
    if [[ -n $drv ]]; then
        awk -v d="${drv%%.*}" 'BEGIN{exit !(d+0 >= 580)}' \
            && ok "driver $drv (CUDA 13.0 image needs >= 580)" \
            || bad "driver $drv is older than the r580 branch the CUDA 13.0 image needs"
    fi
    pw=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits -i "$GPUS" 2>/dev/null | paste -sd+ | bc 2>/dev/null)
    if [[ -n ${pw:-} ]]; then
        if awk -v p="$pw" 'BEGIN{exit !(p > 1100)}'; then
            note "combined GPU power limit ${pw%.*} W. alesha-pro ran 4x3090 at 220 W each; consider nvidia-smi -pl 250"
        else
            ok "combined GPU power limit ${pw%.*} W"
        fi
    fi
fi

hdr "interconnect"
if command -v nvidia-smi >/dev/null; then
    topo=$(nvidia-smi topo -m 2>/dev/null | head -8)
    [[ -n $topo ]] && printf '%s\n' "$topo" | sed 's/^/        /'
    if printf '%s' "$topo" | grep -qE 'NV[0-9]'; then
        ok "NVLink present between at least one pair"
    else
        note "no NVLink. TP4 all-reduces go over PCIe; the 2x170hx run had no P2P at all and still served, but expect this to be the TP4 bottleneck"
    fi
    gens=$(nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.width.max --format=csv,noheader -i "$GPUS" 2>/dev/null)
    printf '%s\n' "$gens" | sed 's/^/        max link: /'
    printf '%s\n' "$gens" | grep -qE '^[12],' && note "a card negotiates Gen1/Gen2; x4-fused risers quarter the PLE host->device traffic"
fi

hdr "host RAM"
# One node-local PLE worker owns the FP8 n-gram table no matter what TP is
# (gpu_worker.py spawn_ple_offload: rank 0 only), so this does NOT scale with
# TP4. Measured 48.4 GiB for the worker, 61.2 GiB for the whole container at
# TP2; TP4 adds two more GPU worker processes, call it 65-68 GiB.
tot_g=$(awk '/MemTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)
av_g=$(awk '/MemAvailable/{printf "%.1f", $2/1048576}' /proc/meminfo)
printf '        MemTotal %s GiB, MemAvailable %s GiB\n' "$tot_g" "$av_g"
awk -v a="$av_g" 'BEGIN{exit !(a >= 70)}' && ok ">= 70 GiB available" \
  || { awk -v a="$av_g" 'BEGIN{exit !(a >= 62)}' \
       && note "${av_g} GiB available; the TP2 container measured 61.2 GiB and TP4 adds two worker processes" \
       || bad "${av_g} GiB available is below the 61.2 GiB the container needed at TP2"; }
sw=$(awk '/SwapTotal/{printf "%.1f", $2/1048576}' /proc/meminfo)
awk -v s="$sw" 'BEGIN{exit !(s < 8)}' && note "SwapTotal ${sw} GiB; the PLE table is touched every step, so swapping it is fatal to throughput"

hdr "disk"
for d in "$MODELS_DIR" "$(dirname "$MODELS_DIR")" /var/lib/docker; do
    [[ -d $d ]] || continue
    free_g=$(df -BG --output=avail "$d" 2>/dev/null | tail -1 | tr -dc 0-9)
    printf '        %-40s %s GiB free\n' "$d" "${free_g:-?}"
done
note "first build peaks at ~335 GB: 180 base + 74 heretic-2 + 51 derived PLE + 29 image"

hdr "docker and overlays"
docker info >/dev/null 2>&1 && ok "docker reachable" || bad "docker not reachable"
docker info 2>/dev/null | grep -q nvidia && ok "nvidia container runtime registered" \
  || note "nvidia runtime not listed by docker info; --gpus may still work via CDI"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    ok "image $IMAGE present"
    al=$(docker run --rm --entrypoint python3 "$IMAGE" -c \
         'import torch;print(" ".join(torch.cuda.get_arch_list()))' 2>/dev/null | tail -1)
    [[ $al == *sm_86* ]] && ok "image has sm_86 kernels (torch arch list: $al)" \
                         || bad "image arch list lacks sm_86: $al"
else
    note "image $IMAGE not pulled yet (scripts/setup.sh)"
fi
if [[ -d $RECIPE/.git ]]; then
    rev=$(git -C "$RECIPE" rev-parse HEAD 2>/dev/null)
    [[ $rev == "$OVERLAY_REV" ]] && ok "overlays at pinned $OVERLAY_REV" \
                                 || bad "overlays at $rev, expected $OVERLAY_REV"
    for f in overlays/ple_layer.py overlays/ple_offload/worker.py overlays/gpu_worker.py \
             overlays/multiproc_executor.py overlays/qwen_gdn_linear_attn.py \
             overlays/shared_experts.py overlays/qwen38/qsa.py overlays/qwen38/ops_qsa.py; do
        [[ -f $RECIPE/$f ]] || bad "overlay missing: $RECIPE/$f"
    done
else
    note "overlays not cloned yet (scripts/setup.sh)"
fi
[[ -f $REPO/results/fp8-kv-scales-tp4-mtp4.json ]] && ok "FP8 QSA scales present" \
  || bad "missing results/fp8-kv-scales-tp4-mtp4.json (scripts/merge-qsa-scales.py)"

hdr "what fits on these cards"
if [[ -n ${smallest:-} && ${smallest:-0} -gt 0 ]]; then
    python3 "$REPO/analysis/plan-memory.py" \
        --card-gib "$(awk -v m=$smallest 'BEGIN{printf "%.3f", m/1024}')" || true
else
    note "skipped: no card size read"
fi

printf '\n%d failure(s), %d warning(s)\n' "$fail" "$warn"
exit $(( fail > 0 ))
