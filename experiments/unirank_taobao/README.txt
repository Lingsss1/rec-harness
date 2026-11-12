UniRank Taobao: unified modeling research, DIN as initial reference

This adapter uses the pinned official Taobao_Action release and UniRank feature
map/loader APIs. DIN establishes the initial empirical reference. Research
explores whole unified architectures: event/field/task representations,
information exchange, history compression, routing and prediction readouts.
Candidates may replace the complete network. parent_trial records comparison
lineage, with no requirement to inherit the parent's architecture. Coherent
structural changes may be designed together and attributed with subsequent
component, replacement or factorial controls.

The initial DIN local-activation pooling uses unnormalized weights over
previous ad fields; action embeddings enter the pooled values. Four separate
prediction heads supervise click/cart/favorite/purchase. This is an adaptation,
not an exact reproduction of the original single-task DIN paper.

Selection: validation macro AUC=(click AUC+cart AUC+favorite AUC+purchase AUC)/4.
Checkpoint saving, parent/incumbent choice, topic gains, merge eligibility and
final selection share selection-policy.json. Exact ties prefer lower macro
Logloss, fewer parameters, then trial ID. Every task's AUC/Logloss and reference
deltas remain explicit; topic decisions must explain trade-offs. Missing tasks
invalidate the main score. Global auc/logloss aliases mean the four-task macro
values; click_auc/click_logloss name click metrics explicitly. Selection uses
validation only. Historical click-selected runs stay frozen; validation-only
reranking is an audit and does not become a new multitask optimization result.
Auxiliary targets match user/category/brand over the next24h; they
do not measure direct advertisement-ID conversion attribution. Never compare
this benchmark directly to an earlier raw-Taobao run with different samples,
histories or labels. Official user filtering, vocabulary counts and static
snapshots are inherited offline assumptions, not established online features.

History policy must be explicit before formal training:
  official_offline: original prefix/actions. Delayed outcomes of recent history
    may contain feedback not yet available at target time.
  matured_24h: exclude timestamp ties; for history age<=24h retain only
    click/exposure; use full feedback after24h maturity. Conservative, since
    exact delayed event times are absent. This changes official input semantics.

Current run: the user chose the paper protocol (official_offline). The choice
is recorded in user-protocol-choice.json. No maturity masking is applied.
DataLoader uses num_workers=0 because512MiB container shared memory cannot
support the original multiprocess-prefetch recipe; all candidates share this
loading policy. The first failed calibration is preserved under setup-attempts.

Prepare source/config/external task snapshot:
  artifacts/unirank-env/bin/python experiments/unirank_taobao/build.py \
    --root artifacts/unirank-taobao-latest

After policy choice, start the source snapshot (set REC_CTR_PROJECT to the
absolute project path). Replace POLICY with the selected value above:
  REC_CTR_PROJECT=/home/liuzhishan/lhr/rec-harness \
    artifacts/unirank-env/bin/python -u \
    artifacts/unirank-taobao-latest/source/campaign.py \
    --root artifacts/unirank-taobao-latest --history-policy POLICY

The runner waits for verified data, audits train/valid, calibrates DIN20steps,
freezes the experiment protocol and cap, trains full DIN, then starts four
independent auto topics on GPUs0..3. Each uses gpt-6.1-sol/high and the original
generic agent instructions/context pattern. Every6completed rounds an isolated
compact decision chooses continue/switch/merge; positive sources must be kept
as controls when merged. A model's architecture may be rewritten fully.

Budget:48h from dataset-source-manifest.json verified_unix (all data downloaded
and checksummed). Acquisition and other work before that timestamp are retained
as separate preparation costs. Every later preparation step, failed attempt,
training/scoring/agent call counts. Rebuilding/restarting retains the same clock
anchor and deadline. The STOP file cancels the campaign. Original
campaigns and their environments/results remain preserved. Research does not
automatically expose the held-out test.

Artifacts: setup.json, dataset-source-manifest.json (pinned digests),
acquisition-status.json, data-audit.json, gpu-contract-check.json (synthetic,
not a research result), environment-freeze.txt, protocol.json (after choice),
references.json (after DIN), topic-board.json and topics/gpu-N/ (after launch).
Individual trials include model.py, model-config.json, result.json, checkpoint,
source/config/protocol hashes, costs and failure logs. No personal knowledge.

Industry/reference knowledge now contains all15models compared in UniRank:
HiFormer, RankMixer, INFNet, LONGER, OneTrans, Zenith, HyFormer, MixFormer,
TokenMixer, EST, HeMix, UniMixer, TokenFormer, UltraHSTU and SSR. Each method
has mechanism/provenance, differences from DIN, conditions, risks and possible
controls. Pinned benchmark code is readable on demand. Original-paper links are
provided with explicit unread status; they do not imply a complete audit or a
local reproduction. Prior Tencent five-scheme assets are archived outside the
active resource manifest. These methods do not become execution baselines.
Task-specific cards stay external to the unchanged generic Agent instructions.

Contract checks:
  cd experiments/unirank_taobao
  ../../artifacts/unirank-env/bin/python -m unittest test_contract -q

Dashboard:
  env -u LD_LIBRARY_PATH -u LD_PRELOAD /usr/bin/tmux \
    -L ctr-research attach -t ctr-research
