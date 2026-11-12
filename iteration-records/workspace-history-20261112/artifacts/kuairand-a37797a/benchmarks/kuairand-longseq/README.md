# KuaiRand-27K long-sequence benchmark

Compare **HSTU**, **ULTRA-HSTU (next-item adaptation)**, **DARE (ID-only
next-item adaptation)** and **CCFormer (causal ID-only next-item adaptation)** using the FuXi-Linear public experiment configuration:
**1024 history events, 128-dimensional embeddings, 128 sampled negatives,
temperature 0.05, global batch 128, AdamW 0.001, 101 epochs, seed 42**.

Target: Linux / NVIDIA A800. Includes portable PyTorch models, preprocessing,
common training, exhaustive catalog evaluation, checkpoint resume, tests and a
Linux Dockerfile. No real KuaiRand results or A800 performance numbers exist yet.

## One command on Linux / A800

From the extracted checkout, with Python 3.10 (including `venv`/`pip`) and a
CUDA-12.1-compatible NVIDIA driver installed:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/oneclick_a800.sh \
  --data-root /data/kuairand-27k --output /data/kuairand-results
```

This creates `.venv-a800`, installs the pinned CUDA 12.1 PyTorch environment,
checks CUDA and runs a small synthetic CUDA/DDP check for each model, downloads
the official **KuaiRand-27K** archive, checks its published
MD5, extracts only four standard-policy logs, preprocesses them, builds the
1024-history cache, runs **HSTU → ULTRA-HSTU adapted → DARE adapted → CCFormer adapted**, and writes
`summary.md`, `summary.csv`, and `summary.json`. No manual data download is needed.
One visible GPU uses accumulation to retain global batch 128; up to two visible
GPUs are used by default. `--nproc 1` or `--nproc 2` makes this explicit.

Use the **same command and paths** after an interruption. HTTP Range resumes a
`.part` download; a server that ignores Range triggers a clean download restart.
MD5 is checked before extraction. Raw logs, processed CSV, provenance and cached
arrays are rechecked before reuse. Atomic staging prevents a partial extraction,
CSV or cache being treated as complete. Model checkpoints resume at saved epoch
boundaries. If the first epoch never produced a checkpoint, the attempt is
preserved under `interrupted/` and that model starts again. Checkpoints also
retain their epoch metrics so a torn log write can be repaired on resume.
Concurrent pipelines sharing data or results are rejected.

Download source: [official Zenodo release](https://zenodo.org/records/10439422),
`KuaiRand-27K.tar.gz`, MD5 `3e3c799a24e2d23a4d2c757fbf9adf59`.
The archive is about 9.9 GB; only the approximately 23 GB of required logs are
extracted. Start with **at least 50 GiB free** for preparation, plus checkpoint
headroom; 80 GiB or more is recommended. Preprocessing concatenates all logs in
RAM: a host with **64–128 GB RAM** is recommended; full-data peak RAM remains
unmeasured. The script checks the initial free-space floor; it does not reserve
storage or change the protocol to fit available hardware.

Useful options:

```bash
# Use an already configured environment (including inside the Docker image).
bash scripts/oneclick_a800.sh --skip-install --data-root /data/kuairand-27k --output /data/results
# Reuse an existing archive; the same official MD5 is required.
bash scripts/oneclick_a800.sh --archive /downloads/KuaiRand-27K.tar.gz --prepare-only
# Select BF16 consistently for all four models.
bash scripts/oneclick_a800.sh --precision bf16 --nproc 2
```

`PYTHON_BIN=python3.10` selects the bootstrap interpreter; Python 3.9–3.11 is
supported by the launcher. `VENV_DIR` changes its isolated environment location.
`--skip-install` uses `PYTHON_BIN` from the active environment. For an OS missing
venv support, install the matching `python3-venv` package first. OS packages and
NVIDIA drivers are not changed by this script.

**Data protocol:** this entry point explicitly selects the released FuXi code
profile described below. If its counts differ from the paper, it passes
`--allow-data-variant` to all four trainers and records the actual counts,
source hash and `paper_statistics_match=false` in the results. It does not claim
paper-identical preprocessing. Add `--require-paper-statistics` to stop before
training on such a mismatch. Failed stages exit nonzero; `pipeline.json` records
the stage and error. Raw data provenance is in `raw/download.json`, processed
provenance in `processed/sasrec_format.provenance.json`, and the common experiment
dataset manifest in the result directory's `dataset.json`.

Final results include NDCG@10/50, HR@10/50, reference/exact MRR, parameter count,
precision, actual dataset statistics and fingerprints. CSV/JSON also contain
logged training time (maximum rank time per epoch), evaluation time and peak
allocated training GPU memory. These costs cover recorded successful epochs;
they exclude download/preprocessing, failed attempts and checkpoint overhead.
Logs are in `logs/{hstu,ultra_hstu,dare,ccformer}.log`; each model keeps its own checkpoint.
Synthetic CUDA checks are isolated under `preflight-smoke/` and excluded from
the real result table. They validate device execution, not full-size memory use.
The pipeline reports success only when all four fixed-final-epoch results pass
the common-protocol checks.

When upgrading from the earlier three-model package, keep the same `--data-root`
to reuse verified data, and select a **new output directory**, for example
`--output /data/kuairand-results-four-models`. Code/config fingerprints changed;
old checkpoints and old three-model results are not silently accepted as this
four-model comparison. Within this version the original command resumes all
four runs as before.

## Install

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-linux.txt
pip install --no-deps -e .
```

The CUDA wheel needs a compatible NVIDIA driver. FP32 is the default;
`--precision bf16` is optional and must be consistent across the four runs.
A800 uses FP32/BF16. There are no H100 FP8 kernels, FlashAttention-3, INT4 serving
or proprietary infrastructure in this package.

## Prepare data

Prefer a verified author-produced `sasrec_format.csv` with `user_id`,
`sequence_item_ids`, `sequence_ratings`, `sequence_timestamps`: chronological,
comma-separated sequences, with zero-based contiguous mapped item IDs.

```bash
python -m longseq_bench.prepare \
  --processed-csv /data/kuairand-27k/sasrec_format.csv --cache /data/kuairand-l1024
```

To generate the CSV from the four KuaiRand-27K **standard-policy** log files:

```bash
python -m longseq_bench.prepare \
  --raw-dir /data/KuaiRand-27K/data \
  --processed-csv /data/processed/kuairand-27k/sasrec_format.csv \
  --cache /data/kuairand-l1024
```

This uses the released-code profile. It requires substantial host RAM: three
columns of all four logs are concatenated in memory. Raw file hashes, output
hash and pandas version are recorded. Existing outputs are not overwritten.

**Paper/code discrepancy:** the paper describes positive-feedback filtering
and an item threshold of 100; released code uses >=350 and iterates a >=20 user
filter. It computes a positive mask on `df` but continues using `df1`, assigned
before filtering. We intentionally preserve that released-code behavior.
The paper lists 27,284 users, 131,090 observed items and 97,010,279 interactions;
the reference candidate catalog enumerates 131,091 IDs. This benchmark preserves
that catalog, including the extra unobserved ID.

Cache metadata records the observed statistics and source hash. A mismatch with
paper statistics requires explicit `--allow-data-variant` when invoking the trainer directly and is
retained in run metadata. Matching counts alone do not establish preprocessing
identity. Code preprocessing is not claimed to regenerate the paper's exact data.

## Run on A800 with an existing cache

Two GPUs, local batch 64 per GPU, are the primary reference configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_a800.sh /data/kuairand-l1024 runs/a800 2 fp32
```

Models run serially, automatically resuming an existing `last.pt` in its original
directory. One GPU is supported by passing `1`: two local batches are accumulated
to retain global batch 128. Order/dropout/partial-last-batch behavior is not
bitwise equivalent to two-GPU DDP. Memory capacity and throughput at batch 64
still need validation on the target A800; an OOM does not silently reduce batch
size or sequence length.

Individual run / resume:

```bash
torchrun --standalone --nproc-per-node=2 -m longseq_bench.train \
  --model ultra_hstu --cache /data/kuairand-l1024 --output runs/ultra_hstu

torchrun --standalone --nproc-per-node=2 -m longseq_bench.train \
  --model ultra_hstu --cache /data/kuairand-l1024 --output runs/ultra_hstu \
  --resume runs/ultra_hstu/last.pt
```

`configs/benchmark.json` is the common executable configuration.
`configs/hstu.gin` preserves the original author's settings for inspection.
Runs record code/data hashes, runtime, precision, parameter counts, per-rank
training time, peak allocated training memory and evaluation time. Resume checks
data/config/code/runtime identity and restores optimizer and per-rank RNG states.

## Implementations and adaptations

| Model | Implemented behavior | Scope |
|---|---|---|
| HSTU | Four STU blocks, SiLU attention, relative time/position bias, U-gating and residual output | Portable tiled PyTorch port of the pinned reference; no production kernels |
| ULTRA-HSTU adapted | SLA local 256/global 32, three long-history layers plus one 512-history layer | Chosen benchmark settings, not undisclosed Meta production settings |
| DARE adapted | Separate 128-D attention/representation tables; candidate-conditioned Top-20; target/history/Hadamard features; 200/80 MLP and two-logit margin | ID-only next-item adaptation; no category/time features, auxiliary CTR loss or batch norm |
| CCFormer adapted | Three directed field-attention flows, causal relative time/position mixing, per-channel subspace SwiGLU, hierarchical causal Conv1D | Independent causal ID-only next-item adaptation; 4 blocks, 128-D, group 8/channel 16, kernel 3/stride 2 |

ULTRA's last layer is computed per causal prefix during all-position training.
Exactly **one** post-truncation layer makes its moving-window computation
equivalent to physically truncating each prefix before that layer. Inference
physically truncates each user's history. Global anchors rebase to each retained
segment. Multiple post-truncation layers are rejected rather than approximated
with an inequivalent shared sliding window. Stochastic length sampling is
disabled to retain the reference supervision; MoT is not used.

DARE conditionally scores every catalog candidate, including all 128 negatives
at **every** supervised position. Tiling/checkpointing bounds memory without
shortlisting or sampling away supervision positions. Representation items and
aggregate history are L2-normalized. The common sampled-softmax replaces its
original CTR loss. It has no candidate-independent two-tower substitute and can
be substantially slower than HSTU in exhaustive evaluation. Transformer-only
configuration fields (blocks/heads/dropout) do not describe DARE's MLP.

### CCFormer adaptation contract

`ccformer` is an independent implementation from
[CCFormer v1, Sections 3.1–3.4](https://arxiv.org/html/2607.28070v1#S3).
The original paper evaluates CTR on Taobao/KuaiRec (default history 200) and
industrial data (default 1000), with rich feature fields and task-specific heads.
This benchmark uses the same FuXi next-item labels, **1024 history, 128-D,
4 blocks, 4 heads, 128 negatives per supervised position, temperature 0.05,
global batch 128, AdamW 0.001, 101 epochs and final-epoch exhaustive evaluation**
as the other models. It does not reproduce the original CTR/AUC results.

| Mechanism | This implementation and difference from the paper |
|---|---|
| User/history/target fields | Two user tokens initialized from each prefix's mean and latest history embedding, plus learned field seeds; history and candidate item IDs share an embedding table. No user ID table, category features or privileged target labels. |
| Cross-field attention | Preserve user→history, target→history and target→user scaled softmax attention. User/history state is shared across candidates; candidates are queries only and never attend to each other. |
| Time/position subspaces | Preserve local non-overlapping relative-time decay and relative-position mixing; use seconds from millisecond logs, positive softplus parameters and a stable decay exponent. Only historical timestamps are consumed. |
| Subspace PFFN | Preserve independent per-channel gated projections over `m × n` vectors, with `m=8`, `n=16`, chosen hidden expansion 2 and SiLU. Add triangular token masks to **both** input projections and the output projection. Time/position mixing is also triangular. |
| Hierarchical compression | Kernel 3/stride 2 Conv1D with left padding. Output anchors are original positions `0,s,2s,…`; only anchors at or before the query prefix are visible. A prefix ending between anchors uses the available coarse tokens while its latest event is still visible in earlier blocks and user initialization. |
| Residuals/head | Chosen pre-norm RMSNorm, residual FFN/mixer/attention paths, two-times-dimensional SwiGLU hidden layers, and an L2-normalized user/pooled-history/target concatenation with a scalar `4d→2d→1` SiLU MLP. These are explicit adapter choices, including the FFN residual topology, and replace the original multi-task prediction heads. |
| Training and evaluation | Common sampled-softmax replaces CTR loss. Every valid next-item position retains all 128 negatives; full catalog scoring is candidate-conditioned, with no retrieval shortlist or two-tower approximation. No BatchNorm or cross-candidate reductions. |

Causal masking and left-padded compression are deliberate changes: unrestricted
mixing within a group would reveal later items under all-position supervision.
The shared-history encoder is checked against physically cropped prefixes,
including partial groups and stride boundaries. It also has zero future-item
gradients and invariant scores under future item/time changes in the tests.
Padding and intermediate group padding are masked after sequence updates.

History/user keys and values are reused across candidate tiles. Query tiling and
activation checkpointing bound memory; target processing still runs over every
candidate at every supervised prefix and can be expensive under this evaluation
protocol. No original-paper speedup or A800 memory-fit claim is made. `run.json`
records the actual architecture and adaptation, and all summary formats label it
**CCFormer-adapted (causal ID-only next-item)**. The original industrial
Numerous-Torch, quantization, hashing and production feature pipeline are absent.

Transformer attention normalization retains the reference's `1024 + 11` reserved
output length. Relative time bias uses the **next interaction's timestamp**,
including the held-out target timestamp, as the reference task does; future item
identities are masked. FP32 log-softmax stabilizes the common loss, whereas the
original trainer hardcodes BF16 interaction scoring on CUDA. This numerical
difference is documented: training trajectories are not claimed bitwise identical.
This normalization paragraph describes HSTU/ULTRA-HSTU; CCFormer uses its own
scaled-softmax field attention and historical-time subspaces as specified above.

## Evaluation

- Per-user leave-last-out, matching `DatasetV2`. Training excludes the last
  interaction and supervises every valid next-item position in the window.
- No random or global temporal split. No separate validation set in the
  reference. Report the **fixed final epoch**, not a test-selected checkpoint.
- Exhaustive shared-catalog scoring; filter input-history IDs, including
  repeated held-out targets, as the reference does.
- NDCG@10/50, HR@10/50 and reference-style MRR: a rank >2500 or a filtered target
  is assigned rank 2501. Also report `mrr_exact` using full ranks and zero credit
  for filtered targets, clearly distinguished from the reference convention.
- Exact score ties use ascending item ID; the original backend's `torch.topk`
  tie behavior may differ. All four models here use the same convention.
- Evaluation tiles only bound memory. Every user and catalog candidate is scored.

## Correctness checks

These use synthetic sequences and are not recommendation results:

```bash
python scripts/fetch_sources.py  # Fetch the pinned reference for parity tests
python -m unittest discover -s tests -v
for model in hstu ultra_hstu dare ccformer; do
  python -m longseq_bench.train --model "$model" --device cpu --smoke --output "runs/smoke/$model"
done
python scripts/summarize.py runs/smoke --include-smoke
```

Replace `--device cpu` with `--device cuda` for synthetic GPU checks. Tests cover
HSTU encoder/parameter-count and block forward/input/parameter gradient parity with author code, future-item
masking, ULTRA train/inference truncation equivalence, DARE candidate dependence,
short prefixes, CCFormer physical-prefix/causality and checkpoint-gradient checks, collision masking and exhaustive candidate-chunk invariance.
The parity test uses CPU layout adapters for fbgemm; it does not replace attention
math. Those adapters are never imported by the trainer. A800/DDP/full-data
validation remains separate.

Outputs: `final.json`, `run.json`, `metrics.jsonl`, `last.pt` per model.
`scripts/summarize.py RUNS` checks common data/history/catalog/precision/seed/epochs
and writes JSON, CSV and Markdown. It refuses to mix smoke output with a real
benchmark; `--require-complete` also rejects missing model results.

Offline workflow tests use a small archive fixture and mocked GPU/training
dispatch. They exercise HTTP Range, ignored Range, checksum failures, selective
extraction, interrupted preprocessing, restart reuse, output locks and complete
result checks. They do not download the real archive or validate A800 execution.

## Open-source packaging

```bash
python scripts/export_public.py /tmp/longseq-kuairand-bench.tar.gz
```

The allowlisted archive includes source, configs, tests, licenses and CI. It
excludes data, runs, checkpoints, virtual environments and fetched vendor trees.
No GitHub repository is created or pushed by these scripts. See Apache-2.0
`LICENSE` and `NOTICE` for attribution and scope.

References: [FuXi-Linear](https://arxiv.org/abs/2602.23671),
[HSTU](https://arxiv.org/abs/2402.17152),
[DARE](https://arxiv.org/abs/2410.02604),
[ULTRA-HSTU](https://arxiv.org/abs/2602.16986),
[CCFormer](https://arxiv.org/html/2607.28070v1).
