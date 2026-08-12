# 4. Experiments

This section evaluates the two-stage AE-SVDD defense under the protocol described
in the supplied experiment document. The design separates three objects that are
often conflated in federated-learning defenses: the client-selection score, the
AE/SVDD training objective, and the final aggregation weights. No numerical result
is filled in here unless it is produced by a completed JSON result file; all tables
below are reporting templates for the analysis scripts.

The experiments answer the following questions:

- **RQ1:** Does the defense preserve clean global-model utility under different
  modalities and attacks?
- **RQ2:** Can it detect individual attacks and identify each attack family in a
  simultaneous mixed attack?
- **RQ3:** Are the reconstruction-only, compactness-only, and complete two-stage
  mechanisms complementary?
- **RQ4:** How sensitive is the method to its loss coefficient, phase-1 duration,
  and trusted validation-set size?
- **RQ5:** How robust is it to attacker ratio, data heterogeneity, client scale,
  and attack intensity?
- **RQ6:** What is the computational overhead relative to the baselines?

## 4.1 Experimental Setup

### 4.1.1 Datasets and Models

Four tasks cover native grayscale images, color images, and text. Dataset loaders
create one fixed, clean server validation subset before client partitioning; those
samples are not assigned to any client.

| Dataset       | Input and classes             | Global model in the codebase                                 | Purpose                                            |
| ------------- | ----------------------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| MNIST         | 28 x 28 grayscale, 10 classes | Native-resolution LeNet-style classifier (`LeNetClassifier`) | Basic image classification and attack sanity check |
| Fashion-MNIST | 28 x 28 grayscale, 10 classes | Lightweight two-convolution CNN (`FashionCNN`)               | Fine-grained grayscale classes and stability check |
| CIFAR-10      | 32 x 32 RGB, 10 classes       | CIFAR-adapted ResNet-18 (3 x 3 stem, no initial max-pool)    | Deeper visual model and the main robustness task   |
| AG News       | Tokenized text, 4 classes     | Lightweight Transformer classifier with a BN compatibility head | Cross-modal transfer to a text model               |

Absolute accuracy is compared only within a dataset. Cross-dataset conclusions
use relative accuracy drop, detection quality, and overhead rather than raw TACC.
For AG News, target-label poisoning has no image trigger; an image-style ASR is
therefore reported as `N/A`.

### 4.1.2 Federated Learning Settings

The primary protocol is fixed for every defense and attack so that only the
defense or attack factor changes.

| Setting                       | Primary value                                                |
| ----------------------------- | ------------------------------------------------------------ |
| Total clients (K)             | 100                                                          |
| Malicious clients             | 30 (30%; client IDs are used only for post-hoc scoring)      |
| Participation                 | All 100 clients each communication round                     |
| Communication rounds          | 300                                                          |
| Local epochs / batch size     | 1 / 64                                                       |
| Client optimizer              | SGD, momentum 0.9                                            |
| Data partition                | Dirichlet α = 1.0 in the primary matrix; IID is the clean reference condition |
| Trusted server validation set | 50 clean training samples, stratified and withheld from clients |
| Random seeds                  | 42, 43, and 44                                               |
| Runtime                       | JSON-driven pipeline, CUDA when available, deterministic job-level seed |

Task calibration is performed once on clean FedAvg using only TACC. The selected
client optimizer settings are then shared by FedAvg, every baseline, and AE-SVDD:

| Dataset       | `client_lr` | `client_weight_decay` |
| ------------- | ----------: | --------------------: |
| MNIST         |        0.10 |                  1e-4 |
| Fashion-MNIST |        0.10 |                     0 |
| CIFAR-10      |        0.05 |                  1e-4 |
| AG News       |        0.10 |                     0 |

The canonical primary generator (`tools/generate_primary_matrix.py`) produces
624 jobs: (3\times7\times8\times3=504) image jobs and
(5\times8\times3=120) AG News jobs. Image attacks are
`none, lf, gn, sf, lie, bd, mix`; AG News uses
`none, lf, gn, sf, lie` because it has no image trigger. The separate RQ3
mechanism matrix uses the same data protocol but varies the malicious ratio and
mechanism explicitly.

### 4.1.3 Attack Settings

The attack implementations are modular client components under `src/attacks`.
Unless a sensitivity experiment says otherwise, the following values are fixed.

| ID       | Attack level                | Upload or data transformation                                | Primary value                                               |
| -------- | --------------------------- | ------------------------------------------------------------ | ----------------------------------------------------------- |
| None     | Control                     | Ordinary local training                                      | No malicious clients                                        |
| LF       | Data poisoning              | Symmetric label map (y' = C-1-y)                             | Fixed by the task label space                               |
| GN       | Model poisoning             | Replace each floating tensor by a moment-matched Gaussian draw | `gaussian_sigma = 0.3`                                      |
| SF       | Model/update poisoning      | Upload (W_g - s(W_l-W_g))                                    | `sign_flip_scale = 1.0`                                     |
| LIE      | Statistical model poisoning | Craft Δ as μ + zσ from benign updates                        | `lie_z_override = 0.524` in the primary matrix              |
| BD       | Data + model replacement    | Lower-right square trigger, target label, then amplify update | target 0; poison 0.6; trigger 5; value 1.0; replacement 3.0 |
| Mix (M1) | Simultaneous mixed attack   | Deterministic round-robin assignment across malicious clients | `lf,bd,gn`                                                  |

Mix is simultaneous: different malicious clients apply different attacks in the
same round. It is not a random choice of one attack per round. The assignment
map is written to each result file, allowing per-family recall. Supplementary
mixed combinations are M2=`lf,sf,lie` and M3=`lf,bd,gn,lie`.

AG News uses target-label poisoning for BD-like behavior if explicitly studied,
but the primary AG News matrix excludes `bd` and `mix`.

### 4.1.4 Baselines

All methods receive the same client states, data partitions, round budget, and
information boundary. No baseline is given the malicious identity or the clean
validation labels unless its protocol explicitly uses the shared validation set.

| Method           | Code ID              | Core operation                                              | Primary 624-job matrix                              |
| ---------------- | -------------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| FedAvg           | `avg`                | Uniform aggregation of all uploads                          | Yes                                                 |
| Trimmed Mean     | `tm`                 | Coordinate-wise trimmed aggregation                         | Yes                                                 |
| Multi-Krum       | `mk`                 | Distance-based Byzantine selection                          | Yes                                                 |
| LASA             | `lasa`               | Layer-adaptive sparsified aggregation                       | Yes                                                 |
| FedSECA          | `seca`               | Sign election and coordinate aggregation                    | Yes                                                 |
| BNGuard          | `bnguard`            | Robust BN-feature distance filtering                        | Yes                                                 |
| FedDMC-style     | `dmc`                | Magnitude, direction, sign, sparsity, and temporal views    | Yes                                                 |
| AE-SVDD (ours)   | `svdd`               | Fixed descriptor, AE reconstruction, and latent compactness | Yes                                                 |
| FL-Defender      | `fld`                | PCA/reputation-based update detector                        | Registered; supplementary matrix only               |
| AlignIns         | `alignins`           | Direction and principal-sign alignment                      | Registered; supplementary matrix only               |
| FLGMM / FLANDERS | `flgmm` / `flanders` | Registered comparison implementations                       | Supplementary only when run under the same protocol |

The primary comparison table must not silently mix supplementary runs with the
624-job matrix. A supplementary defense is included only when its JSON contains
the same task, seed, rounds, client population, attack, and validation protocol.

### 4.1.5 Evaluation Metrics

#### Global utility

- **TACC:** clean test accuracy of the final global model.
- **Accuracy Drop:** (\mathrm{TACC}_{\mathrm{FedAvg,clean}}-\mathrm{TACC}),
  reported on the same dataset and seed.
- **ASR:** for image backdoors, the fraction of triggered test images predicted
  as the target label. TACC and ASR are always reported together for BD/Mix.

#### Client detection

Let malicious clients be positives and rejected clients be predicted positives.
We report detection accuracy (DAR), detection precision (DPR), malicious recall
(RR, equivalent to TPR), benign false-rejection rate (FPR), and F1. The saved
continuous selection scores additionally yield AUROC and AUPRC. AUC is `N/A`
when a slice contains only one class.

For mixed attacks, RR is reported both overall and separately for LF, BD, GN, SF,
and LIE according to the saved client-to-attack map. Overall RR must not hide a
failure on one attack family.

#### Aggregation and stability

We also retain accepted-client fraction, selected rejection ratio, validation
accuracy for each candidate, center norm, latent variance, center shift, and the
three losses. These diagnostics explain whether utility changes come from correct
filtering, excessive benign rejection, or unstable AE training.

Every configuration runs three seeds. Tables report mean +/- sample standard
deviation; the main endpoint is the mean over the last 10 rounds. The final-round
value, best-round value, and complete per-round curve remain available for audit.

### 4.1.6 Implementation Details

AE-SVDD describes the trainable model delta with a fixed hierarchical multi-view
descriptor. The primary descriptor has dimension 4096, seed 2027, and view ratios
global/layer/statistics = 0.5/0.375/0.125. Its input is the pre-round client
delta; BN/LN-specific features are not used by the primary protocol.

The two phases use independent selection-score fields:


$$
r_i = \operatorname{mean}_j |\hat{x}_{ij}-x_{ij}|,
\qquad
d_i = \lVert z_i-c\rVert_2^2.
$$
The primary schedule is `phase1_rounds = 15`, Phase 1 score=`recon`, and Phase 2
score=`svdd`. In Phase 1 the AE is trained with reconstruction loss. In Phase 2
the training objective is
$$
L_2 = \alpha L_{\mathrm{SVDD}} + (1-\alpha)L_{\mathrm{recon}},
\qquad \alpha=0.5.
$$


`alpha` is a loss coefficient only. It does not weight the client-selection score
and it does not change final aggregation weights. The `combined` score, used only
in the score ablation, is the average of the rank-normalized (r_i) and (d_i).

The remaining fixed AE/SVDD values are `latent_dim=64`, `ae_lr=1e-3`,
`ae_weight_decay=1e-6`, `ae_grad_clip=1.0`, `center_ema_decay=0.9`,
`center_init_quantile=0.5`, `phase2_recon_quantile=0.8`, and
`svdd_feature_clip=10`. These are held constant unless a subsection explicitly
varies one of them.

At every phase, the server ranks finite client scores and evaluates the internal
candidate rejection ratios `(0.00, 0.10, 0.20, 0.30, 0.40)` on the fixed clean
validation set. The candidate with the highest validation TACC is selected; an
exact tie chooses the larger rejection ratio. The selected accepted mask is
normalized into the aggregation weights. The candidate grid is an internal
selection protocol, not a user-facing rejection-ratio hyperparameter.

The primary matrix leaves client gradient and update clipping disabled (`None`) so
attack comparisons preserve the canonical upload behavior. A separate numerical-
stability variant may set both `client_grad_clip` and `client_update_clip` to 5.0;
its results are labeled and never pooled with the unclipped primary matrix. AE and
SVDD optimizer clipping remain enabled at 1.0.

All runs are launched from JSON files. Each completed result contains `meta`, an
exactly `total_rounds`-long `rounds` array, effective configuration, attack
metadata, per-round selection scores, accepted IDs, normalized aggregation
weights, candidate validation accuracies, and losses. Truncated or non-finite
results are excluded before aggregation.

## 4.2 Overall Defense Performance and Malicious Client Detection

This combined section reports global utility and malicious-client detection on
the same dataset, partition, attack, defense, and seed slices. The two tables are
kept as separate LaTeX deliverables because utility and detection answer different
evaluation questions. The Markdown document contains only one placeholder per table; no numerical result is inserted here until it is produced by a complete, parseable result file.

### 4.2.1 Overall Defense Performance

The primary matrix compares all eight canonical defenses on every supported attack
and dataset. Results are grouped by dataset and partition; `none` is retained as a
clean control. TACC is reported for every condition. For image BD and Mix, ASR is
reported beside TACC; ASR is `N/A` for AG News and for non-backdoor attacks.

**Table 1. Overall clean utility and attack suppression (mean +/- std).**

**LaTeX placeholder:** `\input{tables/table1.tex}`

The vertical LaTeX table places attack/metric conditions in rows and defenses in columns
so that it remains readable in portrait orientation; it uses a fixed-width `tabular` layout scaled to the text width. Report a separate convergence
plot for representative CIFAR-10 and AG News conditions, with seed
standard-deviation bands. Do not infer a defense win from an accuracy collapse
that happens to reduce ASR.

### 4.2.2 Malicious Client Detection

Detection is evaluated independently from global utility. The accompanying LaTeX
table follows the grouped layout of the supplied reference figure: each attack
family has DAR, DPR, and RR subcolumns, with dataset blocks and detector rows.
The table uses only the attack families defined by this document: LF, GN, SF, LIE,
BD, and simultaneous Mix (M1). AG News has no image-trigger condition, so its BD
and Mix cells are `N/A`.

**Table 2. Malicious-client detection under individual and mixed attacks.**

**LaTeX placeholder:** `\input{tables/table2.tex}`

For individual attacks, report the score distribution over benign and malicious
clients and the per-round decision metrics. For Mix, report overall RR and the
attack-family recall obtained from the saved client-to-attack assignment map.
The complete result analysis also retains FPR, F1, AUROC, AUPRC, TACC, ASR where
applicable, accepted fraction, candidate validation accuracies, and losses; the
compact Table 2 is not used to discard those diagnostics.

## 4.3 Ablation Study

All ablations keep the task, partition, seed, client count, attack parameters,
validation size, and local training budget fixed. Only the named factor changes.

### 4.3.1 Two-stage Architecture

The mechanism ablation compares the following configurations. The phrase
“compactness error” means latent SVDD distance (d_i), not a residual of the
FedAvg aggregation operation.

| Configuration | Phase 1                         | Phase 2                          | Training objective              | Purpose                                  |
| ------------- | ------------------------------- | -------------------------------- | ------------------------------- | ---------------------------------------- |
| FedAvg        | No filtering                    | No filtering                     | N/A                             | Attack-only control                      |
| P1-only       | All rounds ranked by (r_i)      | Not entered                      | (L_{recon})                     | Reconstruction mechanism boundary        |
| P2-only       | Skipped (`phase1_rounds=0`)     | All rounds ranked by (d_i)       | α(L_{SVDD})+(1-α)(L_{recon})    | Early-center and no-warmup behavior      |
| Full          | First 15 rounds ranked by (r_i) | Remaining rounds ranked by (d_i) | Phase-specific objectives above | Complementarity of the complete schedule |

Use the RQ3 runner with LF, GN, SF, LIE, BD, and M1, ratios 10%, 20%, 30%, and
40%, plus a 0% clean control, and seeds 42--44. The default RQ3 runner uses 100 rounds for the
mechanism screen; a selected configuration is confirmed for 300 rounds. Claim
complementarity only when Full improves the same attack/ratio/seed comparison,
not from a cross-condition average.

**Table 3. Two-stage architecture ablation.**

| Malicious ratio | Attack | Configuration | TACC |  ASR |   RR |  FPR | AUROC | AUPRC | Accepted fraction |
| --------------- | ------ | ------------- | ---: | ---: | ---: | ---: | ----: | ----: | ----------------: |
|                 |        |               |      |      |      |      |       |       |                   |

### 4.3.2 Phase-2 Detection Score

Keep Phase 1 fixed to reconstruction, `phase1_rounds=15`, and α=0.5. Compare the
three supported Phase-2 scores:

- `recon`: rank by reconstruction error (r_i);
- `svdd`: rank by latent compactness distance (d_i) (primary);
- `combined`: average the rank-normalized (r_i) and (d_i).

This is a score ablation, not a loss ablation. The loss remains the same for all
three rows. Use 100-round screening over GN, SF, LIE, BD, and M1, then repeat the
best mode on the 300-round primary protocol.

**Table 4. Phase-2 score ablation.**

| Dataset | Attack | Phase-2 score           | TACC |  ASR |   RR |  FPR | AUROC | AUPRC | Selected rejection ratio |
| ------- | ------ | ----------------------- | ---: | ---: | ---: | ---: | ----: | ----: | -----------------------: |
|         |        | recon / combined / svdd |      |      |      |      |       |       |                          |

### 4.3.3 Validation-driven Top-K Selection

The server does not expose a rejection ratio as a tunable run parameter. Instead,
each round it evaluates the internal candidate set

\[
\rho \in \{0, 0.10, 0.20, 0.30, 0.40\},
\]

where ρ is the fraction rejected after ranking finite scores. For each candidate,
the server aggregates the corresponding lowest-score uploads and measures clean
validation TACC. The largest validation-TACC candidate is used for the actual
global update; ties are resolved toward larger ρ. Non-finite rows are ineligible,
even if the nominal candidate would keep them.

To quantify the value of this rule, report (i) the selected ρ distribution, (ii)
the validation accuracy of every candidate, (iii) the fixed-ratio replay for each
candidate, and (iv) the final test TACC/ASR and detection metrics. This directly
tests validation-driven selection without introducing a rejection-ratio
hyperparameter.

**Table 5. Validation-driven Top-K ablation.**

| Dataset | Attack | Selection rule                     | Candidate ρ | Validation TACC | Test TACC |  ASR |   RR |  FPR |
| ------- | ------ | ---------------------------------- | ----------: | --------------: | --------: | ---: | ---: | ---: |
|         |        | validation-selected / fixed replay |             |                 |           |      |      |      |

## 4.4 Parameter Sensitivity

Sensitivity experiments vary one AE-SVDD factor at a time. The primary fixed
configuration is descriptor dimension 4096, Phase 1 length 15, Phase-1 score
`recon`, Phase-2 score `svdd`, and α=0.5 unless the subsection changes that factor.
The screening budget is 100 rounds; the selected setting is re-run for 300 rounds
with seeds 42--44 before being used in the primary claims.

### 4.4.1 Loss Coefficient

Sweep `alpha` in `{0.25, 0.50, 0.75}`. The endpoints 0 and 1 may be included as
diagnostic pure-reconstruction and pure-SVDD controls, but are not needed for the
main sensitivity claim. Keep selection scores fixed (`recon` then `svdd`) so this
experiment changes only the Phase-2 training objective.

**Table 6. Loss-coefficient sensitivity.**

| Dataset | Attack |    α | TACC |  ASR |   RR |  FPR | AUROC | AUPRC | SVDD loss | Recon loss |
| ------- | ------ | ---: | ---: | ---: | ---: | ---: | ----: | ----: | --------: | ---------: |
|         |        |      |      |      |      |      |       |       |           |            |

### 4.4.2 Phase-1 Duration

Sweep `phase1_rounds` in `{5, 15, 30, 50}` with total rounds fixed at 100 for
screening. Phase 1 always uses reconstruction ranking and Phase 2 always uses
SVDD ranking. This isolates how long the AE has to establish a reconstruction
representation before the center-based score is activated.

**Table 7. Phase-1 duration sensitivity.**

| Dataset | Attack | Phase-1 rounds | TACC |  ASR |   RR |  FPR | Selected switch round | Runtime |
| ------- | ------ | -------------: | ---: | ---: | ---: | ---: | --------------------: | ------: |
|         |        |                |      |      |      |      |                       |         |

### 4.4.3 Trusted Validation Set Size

Vary the direct sample count `server_validation_size` in `{10, 25, 50, 100,
200}`. Each set remains clean, stratified, and withheld from clients. The primary
value is 50 samples; it is not 50 groups multiplied by a batch size. Keep the
candidate rejection grid and tie rule unchanged so only validation reliability
changes.

**Table 8. Trusted-validation-set sensitivity.**

| Dataset | Attack | Validation samples | TACC |  ASR |   RR |  FPR | Candidate-choice agreement | Runtime |
| ------- | ------ | -----------------: | ---: | ---: | ---: | ---: | -------------------------: | ------: |
|         |        |                    |      |      |      |      |                            |         |

## 4.5 Robustness Analysis

Robustness factors are environment or attacker factors, not AE-SVDD training
hyperparameters. Unless stated otherwise, use the 300-round primary schedule and
report the final-10-round mean with three seeds.

### 4.5.1 Malicious Client Ratio

Use malicious ratios 10%, 20%, 30%, and 40%, with a 0% clean control. Keep
`num_clients=100` and adjust `num_malicious` so the benign-majority assumption is
explicit. A 50% boundary point may be shown separately, but it is not part of the
main robustness average or a claim under the benign-majority assumption.

**Table 9. Malicious-ratio robustness.**

| Dataset | Attack | Malicious ratio | TACC |  ASR |   RR |  FPR | AUROC | AUPRC |
| ------- | ------ | --------------: | ---: | ---: | ---: | ---: | ----: | ----: |
|         |        |                 |      |      |      |      |       |       |

### 4.5.2 Data Heterogeneity

Compare IID (`dirichlet_alpha=null`), Dirichlet α=1.0, α=0.5, and α=0.1 with
100 clients and 30% malicious clients. Report benign FPR separately from malicious
RR to reveal whether normal client drift is being rejected.

**Table 10. Data-heterogeneity robustness.**

| Dataset |           Dirichlet α | Attack | TACC |  ASR |   RR |  FPR |
| ------- | --------------------: | ------ | ---: | ---: | ---: | ---: |
|         | IID / 1.0 / 0.5 / 0.1 |        |      |      |      |      |

### 4.5.3 Number of Clients

Set (K\in\{50,100,200\}), keep the malicious ratio at 30%, and scale the
malicious count with K. Keep the local batch and round budget unchanged. Report
both quality and wall-clock scaling; do not attribute a change caused by a
different attacker ratio to client scale.

**Table 11. Client-scale robustness and scaling.**

| Clients K | Malicious clients | Attack | TACC |  ASR |   RR |  FPR | Runtime | Peak memory |
| --------: | ----------------: | ------ | ---: | ---: | ---: | ---: | ------: | ----------: |
|           |                   |        |      |      |      |      |         |             |

### 4.5.4 Attack Intensity

Vary one attack-specific strength at a time while keeping all other settings
fixed. The following grid gives interpretable weak-to-strong points:

| Attack | Parameter                      | Values               |
| ------ | ------------------------------ | -------------------- |
| GN     | `gaussian_sigma`               | 0.1, 0.3, 0.5, 1.0   |
| SF     | `sign_flip_scale`              | 0.25, 0.5, 1.0, 2.0  |
| LIE    | `lie_z_override`               | 0.2, 0.524, 0.8, 1.0 |
| BD     | `backdoor_poison_ratio`        | 0.2, 0.4, 0.6, 0.8   |
| BD     | `backdoor_model_replace_scale` | 1.0, 2.0, 3.0, 5.0   |

For each attack, report attack success (ASR where applicable), TACC, RR, and FPR
on an independent horizontal axis. This distinguishes an attack that never
 succeeds from a successful attack that the defense suppresses.

**Table 12. Attack-intensity robustness.**

| Dataset | Attack | Intensity parameter | Value | TACC |  ASR |   RR |  FPR |
| ------- | ------ | ------------------- | ----: | ---: | ---: | ---: | ---: |
|         |        |                     |       |      |      |      |      |

## 4.6 Computational Overhead

Measure overhead on the same hardware, software environment, client count, round
count, and worker schedule. Each measurement excludes dataset download and is
repeated for all three seeds. Report median wall-clock time and standard
deviation; include peak allocated GPU memory and throughput where available.

The comparison separates (i) client local training, (ii) descriptor construction,
(iii) AE/SVDD update, (iv) validation candidate evaluation, and (v) aggregation.
For AE-SVDD, validation evaluation is repeated once per internal candidate ratio;
this cost is part of the method and must not be omitted. JSON and JSONL result
sizes are reported as storage overhead, not GPU time.

**Table 13. Computational overhead.**

| Dataset | Defense | Clients | Rounds | Wall time | Time/round | Peak GPU memory | Throughput (clients/s) | Result size |
| ------- | ------- | ------: | -----: | --------: | ---------: | --------------: | ---------------------: | ----------: |
|         |         |         |        |           |            |                 |                        |             |

All conclusions must be made from complete, parseable result files with the
expected round count. Incomplete runs, non-finite feature failures, OOM traces,
and truncated logs are reported as failures and excluded from mean/std tables;
they are summarized separately in the reproducibility appendix.
