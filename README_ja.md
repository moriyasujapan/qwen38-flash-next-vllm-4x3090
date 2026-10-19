# Qwen3.8-Flash-Next W4A16 を RTX 3090 4枚で

[qwen38-flash-next-vllm-2x170hx](https://github.com/moriyasujapan/qwen38-flash-next-vllm-2x170hx)
を RTX 3090 4枚へ移植したもの。TP4 + expert parallel、同じ heretic-2 W4A16 チェックポイント、
同じローカル生成 FP8 PLE をホスト RAM にオフロード。

> **このリポジトリの内容は 3090 実機で一度も動かしていない。** 設定の検討結果である。
> 数値はすべて (a) 親リポジトリが 2× CMP 170HX で実測したもの、
> (b) [alesha-pro](https://github.com/alesha-pro/qwen38-flash-next-4x3090) が 4× RTX 3090 で
> 実測したもの、(c) それらから `analysis/plan-memory.py` が導出した算術のいずれかで、
> どれなのかは各項目に書いてある。実機ではまず `scripts/preflight.sh` を回すこと
> （実際に見つけたカード容量で収まり表を再計算する）。

---

## 構成を決めてしまう唯一の事実

**TP4 にしてもカード1枚あたりの KV キャッシュは TP2 と全く同じ大きさで、小さくならない。**
pin された image の `vllm/models/qwen3_8_flash_next/nvidia/qsa.py:214-220` より:

```python
self.total_num_kv_heads = int(config.num_key_value_heads)   # 2
if self.total_num_kv_heads >= tp_size:
    if self.total_num_kv_heads % tp_size:
        raise ValueError("QSA KV heads must be divisible by TP size")
elif tp_size % self.total_num_kv_heads:
    raise ValueError("TP size must be divisible by replicated QSA KV heads")
self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

`num_key_value_heads` は 2 なので `num_kv_heads` は `max(1, 2 // tp)` ——
**TP2 で 1、TP4 で 1、TP8 でも 1**。`k_proj`/`v_proj` は `(512, 2560)` = 2 head × 256 で、
2 rank を超える分は分割ではなく複製される。さらに `common/qsa_cache.py:683` が
KV head をちょうど 1 個しか受け付けないので、TP1 は却下、TP は偶数でなければならない。

結果として、2枚→4枚で増えるのは **重み** の置き場だけで、**コンテキスト** は増えない。
重みはきれいに分割され 1枚あたり 37.43 → 19.48 GiB になる。だが 3090 は 24 GiB なので、
余りは 170HX の約 25 GiB ではなく **約 3 GiB** しかなく、トークンあたり KV は
1枚あたり 16,405 B (BF16) / 12,045 B (FP8) で変わらない。262,144 トークンのコンテキスト 1本で
BF16 KV が 1枚あたり 4.00 GiB 必要 —— 収まらないし、カードを増やしても収まらない。

つまりこの移植は「`--tensor-parallel-size 4` にするだけ」ではない。親リポジトリの看板機能
だった **262K コンテキスト** と **MTP 投機デコード** の、どちらかを選ぶ話になる。
両取りはできない（ぎりぎりのプロファイル D を除く）。

## 何が収まるか

`analysis/plan-memory.py`、24 GiB カード、`--gpu-memory-utilization 0.98`、TP4:

| プロファイル | コンテキスト | MTP | KV dtype | KV 空き | KV トークン | 必要 | 判定 |
|---|---:|---|---|---:|---:|---:|---|
| **A** | 262,144 | off | fp8_e4m3 | 3.13 GiB | 279,288 | 2.94 GiB | 収まる 1.07× |
| A' | 262,144 | off | BF16 | 3.13 GiB | 205,061 | 4.01 GiB | **不可** |
| **B** | 131,072 | k=4 | fp8_e4m3 | 1.73 GiB | 154,041 | 1.47 GiB | 収まる 1.18× |
| B+ | 196,608 | k=4 | fp8_e4m3 | 2.45 GiB | 218,759 | 2.21 GiB | 収まる 1.11× — 未作成のチェックポイントが必要 |
| **C** | 98,304 | k=4 | BF16 | 1.73 GiB | 113,101 | 1.50 GiB | 収まる 1.15× |
| C' | 65,536 | k=4 | BF16 | 1.45 GiB | 94,774 | 1.00 GiB | 収まる 1.45× |
| D | 262,144 | k=4 | fp8_e4m3 | 2.82 GiB | 251,297 | 2.94 GiB | 0.98 では**不可**、0.99 で収まる |
| — | 262,144 | k=4 | BF16 | 12.6 GiB | 1,151,760 | 4.01 GiB | 2x170hx の既定値（比較用） |

1.07× は「フル長のリクエスト1本で他に何も流せない」という意味。親リポジトリは 3.16× ある。

### メモリの行き先、1枚あたり

| | 2×170HX (TP2, 実測) | 4×3090 (TP4, 導出) |
|---|---:|---:|
| カード | 63.4 GiB 利用可 | 24.0 GiB |
| 重み（MTP head 込み） | 37.43 | **19.48** |
| non-torch (CUDA ctx, NCCL) | 1.27 | 約 1.30 |
| prefill activation | 5.84 (batched 8192) | 0.73 (batched 1024) |
| CUDA graph + decode バッファ | 1.09 (seqs 8) | 0.14–0.28 (seqs 1–2) |
| **KV に残る分** | **17.60** | **1.73–3.13** |

重みの列は `analysis/weight-split.py`。safetensors のヘッダを直接読み、image から読み取った
分割ポリシーを適用する。その TP2 列は親リポジトリが実測した「重み + non-torch」38.7 GiB を
1.27 GiB の差で再現し、その残差を non-torch 行に使っている。activation の傾きは親リポジトリが
実測した batched tokens 1024 → 8192 のステップ（KV を 5.11 GiB 削る必要があった）から。
このモデルを alesha-pro の 4x3090 実測に当てると FP8 KV 279,288 トークンと予測し、
実測は **276,145** —— 1.1% 高い。

```
analysis/weight-split.py --model-dir <checkpoint> --tp 2 --tp 4
analysis/plan-memory.py  --card-gib 24 --util 0.98
```

## プロファイル

```bash
GPUS=0,1,2,3 PROFILE=a ./scripts/run.sh    # 262K, MTP なし, FP8 KV
GPUS=0,1,2,3 PROFILE=b ./scripts/run.sh    # 131K, MTP k=4,  FP8 KV
GPUS=0,1,2,3 PROFILE=c ./scripts/run.sh    #  98K, MTP k=4,  BF16 KV
```

**A — 長文コンテキスト、MTP なし。** 保守的な選択で、3090 実測の裏付けがある唯一の点
（alesha-pro が検証した動作点そのもの: 262K 設定、FP8 QSA KV、`max-num-seqs 1`、
`max-num-batched-tokens 1024`、util 0.98）。代償は速度で、親リポジトリの実測は MTP オフで
**65.8 tok/s**、k=4 では 101–170 tok/s。MTP を落とすと散文で 1.6 倍、コードで 2.6 倍遅くなる。

**B — MTP を残してコンテキストを半分に。** 親リポジトリのデコード優位がそのまま残る
（k=4 実測で日本語散文 101 tok/s、コード 170 tok/s）。131,072 が算術上の上限
（空きは 154,041 トークン）。170HX 構成から来て対話レイテンシを重視するなら、まずこれ。

**C — MTP を残して FP8 KV を一切使わない。** BF16 KV なので QSA overlay も校正ファイルも
FP8 KV の精度論も不要。98,304 トークン。最初の起動確認に向く最も単純な構成。

**B+ — MTP つきで 196,608。見つけたが未実装。** MTP draft head の MoE エキスパートは
量子化されず **BF16** のまま残っている: `quantization_config.dynamic` が
`{"-:.*mtp.*": {}}` で、`mtp` に一致するものを GPTQ から除外しているため。該当する2本 ——
`mtp.layers.48.mlp.experts.gate_up_proj` `(512,1280,2560)` と `.down_proj` `(512,2560,640)`
—— が draft head 4.86 GiB のうち 4.69 GiB、すなわち **TP4 で MTP が食う 1枚 1.26 GiB のうち
1.17 GiB** を占める。これを FP8 E4M3 で持てば 1枚 0.59 GiB 空き、MTP の上限が 131K → 196K
に動く。

これが「未実装」である理由: 単純な dtype 書き換えは壊れたチェックポイントを作る。モジュール側は
BF16 として読むので FP8 のバイト列はゴミになる —— alesha-pro のビルダが
`shared_expert_gate` 48本を BF16 に戻して回避しているのと同じ罠である。正しくやるには、
その一部モジュールにだけローダが従う設定、つまり GPTQ グループと FP8 グループを併記した
`compressed-tensors` ドキュメントを出す必要があり、一方で image は `gptq` をそのまま
`auto_gptq` に写している。image には `compressed-tensors` / `fp8` / `modelopt_mixed` が
入っている（`QUANTIZATION_METHODS`）ので部品は揃っているが、
`Qwen4ExpForConditionalGeneration` が2スキーム併記を読めるかは未検証。`PROFILE=b+` は
`$MODELS_DIR/Qwen3.8-Flash-Next-heretic2-W4A16-mtpfp8` を見に行き、そこに何か置かれるまで
素直に失敗する。

**D — MTP つきで 262K、ぎりぎり。** 必要なものが全部: FP8 MTP エキスパート (B+)、FP8 KV、
`--max-num-batched-tokens 512`、`--max-num-seqs 1`、**かつ** util 0.99 ——
24 GiB カードに約 240 MiB しか残らない。親リポジトリでは 753 MiB の余裕でも vLLM 自身の
「GPU メモリを使い切る」提案が 8.9K prefill で OOM している。OOM すると思って良い。
算術が閉じるので挙げているだけで、推奨ではない。

**プロファイル A で試す価値のある「VRAM ゼロの投機デコード」。** `ngram`、`ngram_gpu`、
`suffix` はいずれも image の `SpeculativeMethod` リテラルに入っており draft 重みを持たない。
VRAM を一切使わないので A の 262K をそのまま保てる:

```bash
GPUS=0,1,2,3 PROFILE=a SPEC='{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":4,"prompt_lookup_min":2}' \
  ./scripts/run.sh
```

なお QSA リングのサイズは k=1〜4 で同一。`common/qsa_cache.py` が
`capacity = compress_ratio * cdiv(compress_ratio + k, compress_ratio)` を
`compress_ratio = 4` で計算するので k ∈ 1..4 はすべて 8、k = 0 だけが 4 になる。
KV の都合で k を 4 より下げる理由はない。

## 2×170HX 版からの変更点

| 設定 | 2×170HX | 4×3090 | 理由 |
|---|---|---|---|
| `--tensor-parallel-size` | 2 | **4** | — |
| `--enable-expert-parallel` | 必須 | **やはり必須** | `moe_intermediate_size` 640 / 4 = 160 は量子化の group size 128 で割れない。EP なら各 rank がエキスパート単位（512 / 4 = 128）を持ち 640 が保たれる |
| `--max-model-len` | 262144 | **プロファイル次第 98304–262144** | 上記の KV の事実 |
| `--speculative-config` | MTP k=4 | **オフ、またはコンテキストを下げて k=4** | draft 重みが 1枚 1.26 GiB |
| `--kv-cache-dtype` | `auto` (BF16) | **131K 以上では `fp8_e4m3`** | BF16 262K は 1枚 4.00 GiB 必要、空きは 3.13 |
| `--max-num-batched-tokens` | 8192 | **1024** | prefill batch 1024 トークンごとに 1枚 0.73 GiB。あと7本分の余地はない |
| `--max-num-seqs` | 8 | **1–2** | 1 スロットあたり約 0.14 GiB、そもそも KV プールが長いリクエスト8本を持てない |
| `--gpu-memory-utilization` | 0.92 | **0.98** | 余裕を押し切る必要がある。4枚ともヘッドレスで回すこと |
| `--kv-cache-memory` | 18897856102 を明示 | **空（vLLM の profiling に任せる）** | 170HX では実在する余裕から容量を買えた。ここでは余裕が 0.1–0.2 GiB で、vLLM の profiling が安全側。ブロック不足で `MAX_MODEL_LEN` を拒否されたときだけ手で指定する |
| `--pipeline-parallel-size` | 拒否される | **やはり拒否される** | `overlays/gpu_worker.py:238` が `VLLM_PLE_CPU_OFFLOAD` 下で `PP=2` を未対応として挙げている。同様に `nnodes≠1`、非ローカル DP、PCP、DCP、ubatching、weight transfer も不可。TP には制限がない |

それ以外 —— image、overlay、`VLLM_PLE_*`、`VLLM_HTTP_TIMEOUT_KEEP_ALIVE=75`、
`--compilation-config {"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}`、`qwen3_xml` tool parser、
`SYS_PTRACE` —— はそのまま使える。

## そのまま通るもの、およびその確認方法

**overlay は TP サイズに依存しない。** `overlays/ple_offload/worker.py` は
`num_workers == dp_size * tp_size` を検証し target を `tp_rank` で並べるだけで、2 を
ハードコードした箇所はない。`gpu_worker.py` が拒否する並列化は上記のリストだけで、TP は入っていない。

**ホスト RAM は TP で増えない。** これが最大のリスクだった —— FP8 PLE テーブル 48.4 GiB が
rank ごとなら TP4 で 193 GiB になる。そうはならない。
`overlays/gpu_worker.py` の `spawn_ple_offload()` は `self.rank == 0` 以外で早期 return するので、
TP に関係なく **node-local な PLE worker はちょうど1個**。docstring も
"Own all discovered PLE tables and serve every local DP rank" と言っている。
TP rank ごとに増えるのは `max_num_batched_tokens × ple_embed_dim` の pinned scratch buffer
1本だけで、1024 × 2560 なら 5 MiB。TP2 のコンテナが実測した約 61 GiB に worker プロセス
2個分を足して 65–68 GiB と見ればよい。ホスト RAM 96 GiB で余裕、128 GiB が推奨。

**FP8 KV の校正はそのまま TP4 で使える。** 公開されている2つの校正はどちらも層ごとの K/V
absmax を *TP rank 間の最大* として記録しており、TP ≥ 2 では `num_kv_heads` が常に 1 で
`k_proj`/`v_proj` が 2 rank を超えて分割されないため、この量は TP 不変である。数値でも確認した:
親リポジトリの TP2 ファイルと alesha-pro の TP4 ファイルは要素ごとに平均 9% / 最大 33% の差で
一致し、共通する **12 層すべて** で親リポジトリ側が alesha-pro 以上
（元々両者の要素ごと最大として作られている）。`scripts/merge-qsa-scales.py` がマージを再生成し、
`results/fp8-kv-scales-tp4-mtp4.json` がその結果 —— 層ごとの値は親リポジトリのファイルと同一で、
provenance が付いただけ。そしてこれは MTP drafter の attention 層（`mtp.layers.48`）の
エントリを持つ **唯一の** ファイルで、プロファイル B / B+ / D が必要とするのはまさにそれである。

**image に sm_86 カーネルが入っている。** 3090 は sm_86、170HX は sm_80 で、image は独自ビルド
なので確認が必要だった。`torch.cuda.get_arch_list()` は
`sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`、`TORCH_CUDA_ARCH_LIST` は
`7.5 8.0 8.6 8.9 9.0 10.0 12.0`、`cuobjdump --list-elf vllm/_C_stable_libtorch.abi3.so` に
`sm_86.cubin` がある。`preflight.sh` が再確認する。

**head 数は割り切れる。** `num_attention_heads` 24 / 4 = 6。`linear_num_key_heads` 16 / 4 = 4、
`linear_num_value_heads` 48 / 4 = 12。`num_experts` 512 / 4 = 128。`vocab_size` 248320 / 4 = 62080。
vision tower の `num_heads` 16 / 4 = 4、`intermediate_size` 4304 / 4 = 1076。
`shared_expert_intermediate_size` 640 / 4 = 160 は group size 128 の倍数では **ない** が、
shared expert は量子化されていない BF16（GPTQ なのは routed expert の `qweight` 73,728 本だけ）
なので group の制約を受けない。`indexer_n_heads` 4 と `indexer_kv_heads` 1 は無関係で、
`QSAIndexer` は "Replicated Q/K projection" と明記されている。

## スループット: ここでは予測しない

逆向きに効く2つの要因があり、どちらも 3090 では未実測。

- **4×3090 に有利:** 合計メモリ帯域。デコードは active な重みの帯域で律速され、各カードは
  その1/4しか持たない。4 × 936 GB/s の GDDR6X は HBM2e 2枚より合計帯域が大きい。
- **4×3090 に不利:** インターコネクト。TP4 は層ごとの all-reduce が TP2 の2倍になり、
  expert parallel も MoE 層ごとに 4 rank の all-to-all に広がる。170HX のペアは
  **P2P なし**（`can_device_access_peer` が両方向 false）で PCIe Gen2 x16 のまま 100–170 tok/s
  出していたので致命傷ではないが、悪化するのはこの項である。NVLink ブリッジは 3090 を2枚ずつ
  繋ぐので、6本のリンクのうち2本が改善する。

正直な予想は、プロファイル A が親リポジトリの MTP オフ実測 **65.8 tok/s** 付近、
プロファイル B が散文 / コードの **101 / 170 tok/s** 付近、そこにインターコネクト分の増減。
測ること:

```bash
until curl -sf localhost:18024/health; do sleep 10; done
python3 bench/bench.py --reps 3 --prefill 8192,32768 --conc 1,2,4
```

なおプロファイル B の 154,041 KV トークンは **33K のリクエスト4本同時**（132K）を保持できるので、
親リポジトリの「4 並列で 392 tok/s」が明らかに届かないわけではない。4×3090 が失うのは
長い単一コンテキストで、中程度の長さでの並列性ではない。

## クイックスタート

```bash
git clone <this repo> && cd qwen38-flash-next-vllm-4x3090

GPUS=0,1,2,3 ./scripts/preflight.sh   # 読み取りのみ。実機のカードで収まり表を再計算する

./scripts/setup.sh              # pin された overlay + vLLM image (28.8 GB)
./scripts/download-model.sh     # base W4A16, 179.8 GB, 再開可能
./scripts/build-ple-fp8.sh      # BF16 PLE -> FP8 + manifest
./scripts/download-heretic2.sh  # heretic-2 W4A16, 74 GB, sha256 検証
./scripts/prepare-heretic2.sh   # GPTQ config, base tokenizer, MTP head graft

GPUS=0,1,2,3 PROFILE=c ./scripts/run.sh   # まず一番単純なプロファイルで
docker logs -f flashnext-vllm
```

エンドポイントは `http://localhost:18024/v1`、モデル名 `flash-next-w4a16`。
起動は TP2 で約 5 分 50 秒だった。TP4 では CUDA graph capture が rank ごとなのでもう少しかかる。

**`chat_template_kwargs: {"reasoning_effort": "medium"}` を送ること。**
chat template の既定は `xhigh` で、親リポジトリの実測では 6 回のうち 5 回、20,000 トークンの
予算を全部思考に使って回答しなかった。移植で変わっていないし、サーバが壊れているように
見える原因として一番ありがちな設定である。

## 未解決の論点

1. **0.1–0.2 GiB の余裕。** ここに挙げた全プロファイルが、親リポジトリの 753 MiB
   （それでも1段上で OOM した）より薄い。プロファイル A が util 0.98 で起動しないなら、
   コンテキストを削る前に `--max-num-batched-tokens` を 512 に落とす。
2. **TP4 の non-torch は本当に約 1.30 GiB か。** NCCL バッファはグループサイズで増える。
   1.6 GiB だった場合、プロファイル A は KV 約 27K トークンを失うがまだ収まり、B も同様。
   壊れはしないが余裕は減る。`preflight.sh` では測れない。1回起動すれば engine ログの
   メモリプロファイル行で分かる。
3. **この規模での FP8 KV の精度を再確認すること。** 親リポジトリは FP8 KV の検索性能は
   BF16 と同等、デコードは 6–11% 遅いと実測した。ここでは 131K 超では選択肢ではなく必須で、
   しかもその遅さが既に MTP を諦めた構成に乗る。共通 12 層のスケールは TP4 校正由来なので
   移植の中で最も裏付けが強い部分だが、チェックポイントの構成は別物で校正されている。
4. **プロファイル B+ は作れるのか。** 上記のとおり。MTP つきで 196K 以上に行く唯一の道。

## クレジット

親リポジトリのクレジットすべてに加えて:

- [alesha-pro/qwen38-flash-next-4x3090](https://github.com/alesha-pro/qwen38-flash-next-4x3090)
  —— ここで mount している PLE offload overlay、QSA FP8 KV の経路、そしてこの検討が
  校正に使った唯一の 4×3090 実測値。
- [moriyasujapan/qwen38-flash-next-vllm-2x170hx](https://github.com/moriyasujapan/qwen38-flash-next-vllm-2x170hx)
  —— ここでの導出すべての出発点になった実測値。

## ライセンス

このリポジトリのコードは Apache-2.0。重みは含まない（Qwen Community License）。
