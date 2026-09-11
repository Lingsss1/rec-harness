# Shared-work call gradient decomposition

Status: researched proposal; no implementation, validation result, accepted tool or model-edit handoff.

Target: distinguish opposing first/second-call contributions from small total gradients before another round of forward-capacity trials. The final objective remains better second-round retrieval and full validation quality/cost, not merely producing this diagnostic.

## Primary methods considered

1. [PCGrad, NeurIPS2020](https://papers.neurips.cc/paper_files/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf): negative cosine defines task-gradient opposition; the paper explicitly says conflict alone is not necessarily detrimental. We borrow vector geometry for a different quantity—per-invocation contributions to one task loss. We do not implement gradient surgery or change PSGDKron. No paper code reused.
2. [GradNorm, ICML2018](https://proceedings.mlr.press/v80/chen18a.html): gradient magnitudes motivate an existing-tool baseline. Magnitude does not uniquely identify opposite directions. We do not implement adaptive task weighting or change loss weights. No code reused.
3. [Recurrent depth / Huginn, arXiv2502.05171v1, 2025-02-07](https://arxiv.org/html/2502.05171v1): §3.3 backpropagates only through the final k iterations (k=8), while injected entry embeddings still receive gradients. This concerns long random-depth language-model training and does not justify detaching a two-round recommender. Appendix warns of mixed-precision/truncation implementation subtleties. Useful alternative training-path motivation, not the chosen estimator. No code reused; v3 URL unavailable and OpenReview access challenged, so cite this actual v1.

## Measured quantity and scope

At one fixed checkpoint and batch, use separate equal-valued parameter leaves for the two calls to the shared work layer, without detaching its states. Let g1,g2 be derivatives of the same task loss with respect to these leaves. Chain rule gives native tied gradient g=g1+g2 in real arithmetic. Verify it empirically before using any diagnostic output. First-call credit includes its downstream state and prediction paths; g1 is not a separate first-round objective.

Report each named parameter group, task and batch: norms, dot product, cosine, cancellation ratio ||g1+g2||/(||g1||+||g2||), native reconstruction error, positives/negatives and row/user identity. Also report task-summed raw gradients separately; do not confuse across-task and across-round effects. Near-zero norms yield null/abstention.

Original Q/K primitives, probabilities, masks and prediction output must remain the native forward. Instrumentation is temporary and repository-local. A thin independent-leaf proxy with restoration may be used; verify parameters/buffers/RNG before/after. No optimizer step/state repair, checkpoint write, extra training seed, test split or permanent model edit. Base implementation and extra V branch remain frozen.

BF16 shared autocast leaves can accumulate differently than separate leaves: require exact output equality and normalized reconstruction error <=0.02; failure is invalid, not a reason to silently disable native AMP or retry with a wider tolerance. Raw Euclidean cosine does not recover historical HeavyBall preconditioned updates from weight-only checkpoints. Negative cosine alone cannot establish optimization harm, unavailable information, or likely AUC improvement.

## Validation contract

Precommit: recurrence-gradient-credit/thresholds.json. Analytical aligned/orthogonal/opposed vectors and matched-total-gradient controls precede real data. Synthetic toy calls at zero activation can have the same native gradient and activation-times-gradient score but different call contributions. Recover this distinction and return null for zero gradients. Exercise shape/nonfinite/empty/scale/determinism/resource boundaries independently.

Shadow: existing mature_r2 and event_work on the same first four validation batches of8192 rows. At most16 full forwards and64 task backwards, zero updates. Retain original native gradients and matched decompositions in bounded arrays; reuse original input protocol and report all class counts. Under25 task positives or negatives means no task action. No pooling away batch instability.

Utility: compare no-tool AUC/latency decisions, existing norm/activation summaries, and their union with the candidate. Accept only after all six tool gates pass and a predeclared local condition is reliably distinguished or falsified. A null real result may reject the tool; green synthetic tests do not establish practical utility.

Future action eligibility requires query-group opposition with cosine<=-0.2 and ratio<=0.6 in>=3/4 batches in both fixed contexts, plus all fidelity/sufficiency gates. Even then, only a matched sharing experiment with a capacity control may be considered after root review; no optimizer change or automatic AUC/causal claim. No qualifying pattern means no edit from this probe. More generally, report uncertainty rather than retuning cutoffs after seeing data.

Budget: offline<=120s/2GiBRSS; conditional shadow<=900s wall/3600s CPU/16GiBRSS/64GiBVRAM; all artifacts<=512MiB. NumPy analysis should stream by named group. No packages, cloning or installation needed. No stable toolkit files are modified during candidate research.
