# Project A: Scale-up Plan

*Draft, 3 October 2026. Goal: turn the current preregistered study (MIMIC-CXR, n = 100, two or three models) into a multi-dataset, multi-model study with validated scoring and a properly powered confirmatory test.*

---

## 1. What this plan fixes

| Weakness reviewers will raise | Fix in this plan |
| --- | --- |
| One dataset, 100 studies | Three datasets with different patient populations, about 300 target studies each |
| Two or three models | Seven to eight models spanning specialist vs generalist, medical vs general-purpose, and size |
| No expert validation | Scorers validated against existing radiologist annotations (ReXVal), plus a radiology-trained evaluator (GREEN). No in-house clinical judgement needed |
| Primary outcome null | A new, separately preregistered confirmatory study, powered from the current effect sizes, using the scorer that best tracks radiologists |
| (Kept as is) MedGemma in 8-bit, small LLM judge | Disclosed as limitations. The judge stays, but is no longer the only or primary scorer |

The current study (Study 1) is **not re-analysed or re-run**. Its preregistered results stand as reported. The scale-up is a new study (Study 2) with its own preregistration.

---

## 2. Constraints this plan is built around

- **Compute:** one 47 GB MIG slice per account (QOS `student-1mig-limited`), so jobs run one at a time. CPU work runs on the login node, at reduced priority.
- **People:** one undergraduate, with no clinical expertise in-house. Validation must use existing expert-annotated resources.
- **Data governance:** each dataset has its own access terms. Nothing derived from MIMIC, CheXpert or PadChest may be committed publicly, only IDs and aggregates, as now.
- **Contamination:** models may have been trained on these datasets. Test splits are used where they exist, and a contamination table is reported (see section 5).

---

## 3. Phase 0: Validate the scorers first (about 1 week)

The current scorers disagree on one key question (whether Lingshu's real-image report beats its image-free one). Study 2's primary scorer should be chosen **before** any Study 2 data exist, based on how well each scorer agrees with radiologists.

**Scorers:**

1. **LLM judge.** The existing Llama-3.1-8B judge, kept for continuity.
2. **CheXbert label agreement.** Already running.
3. **RadGraph F1.** Entity and relation overlap, including location and laterality. Runs on CPU.
4. **GREEN.** A radiology-trained LLM evaluator that counts clinically significant errors. It needs a GPU but fits the MIG slice.

**Validation data:** ReXVal (PhysioNet, credentialed). Radiologists counted errors in candidate reports for MIMIC-CXR studies.

**Analysis:** for each scorer, compute the rank correlation (Kendall's τ) with radiologists' total and clinically significant error counts. Report all four.

**Decision rule, to be written into the Study 2 preregistration:**

- The scorer with the highest τ against clinically significant errors becomes the **primary scorer** for Study 2.
- The two next-best become secondary scorers.
- Claims of weak image dependence require agreement between the primary scorer and at least one secondary scorer.

**Deliverable:** a scorer-validation table. This is a small, publishable result on its own.

---

## 4. Phase 1: Models (about 1–2 weeks, overlapping Phase 0)

The aim is to cover the dimensions that might drive image dependence, not to maximise the model count. **Availability, licences and input formats must be checked for each candidate before it's committed to.**

| Role | Candidates (verify) | Why |
| --- | --- | --- |
| Chest X-ray specialist | MAIRA-2 ✅, CheXagent, LLaVA-Rad | Positive controls; tests whether specialists consistently use the image more |
| Medical generalist | Lingshu-7B ✅, MedGemma-27B (int8) ✅, MedGemma-4B, HuatuoGPT-Vision | The main object of study |
| General-purpose base | Qwen2.5-VL-7B (Lingshu's base model) | A paired ablation: does medical fine-tuning change image dependence? |
| Size contrast | A larger sibling of one generalist, if it fits in int8 | Does scale change image dependence? |

The target is **7 to 8 models**. The Qwen2.5-VL vs Lingshu pair is especially valuable: same architecture, with and without medical training, which directly tests whether medical fine-tuning teaches the population prior.

**Engineering:** replace the per-model runners with **one config-driven harness**, built from:

- a common runner (conditions, resume, logging, decoding),
- one small adapter per model family. A generic chat-VLM adapter covers Lingshu, Qwen2.5-VL and MedGemma; MAIRA-2 and other structured-input models get their own.

Each model's input adaptations, such as requiring an image or using structured fields, are declared in its config and recorded in every output row.

Each model gets a **smoke test of one study, five conditions** before any full run, as was done for MedGemma.

---

## 5. Phase 1: Datasets (about 2 weeks, overlapping)

Datasets are chosen for **different patient populations**, because the central mechanism, population priors matching the dataset, predicts that image-free reports score well only where the population matches the model's prior.

| Dataset | Population | Access | Role |
| --- | --- | --- | --- |
| MIMIC-CXR (test split) | Hospital/ICU, mostly abnormal, AP-heavy | Have it | Continuity with Study 1. Expand targets toward all eligible test patients |
| IU X-ray (OpenI) | Outpatient, mostly normal | Public | The key contrast for the base-rate mechanism; small and easy to access |
| CheXpert Plus | Different hospital, mixed | Stanford AIMI research agreement | A second institution with full reports |
| PadChest (optional) | Different country; Spanish reports (English translations exist) | BIMCV agreement | Language and population shift; only if time allows |

**Engineering:**

- **A per-dataset adapter** that produces the same manifest format the selection script already uses: study ID, patient ID, frontal image path, view, report text.
- **CheXbert-derived labels for pairing.** Datasets without official labels get them by running CheXbert over their reports, which keeps pair selection identical across datasets.
- **The same selection rules everywhere:** one target per patient, sources from non-target patients, mostly contrast pairs plus some same-label pairs, and a fixed seed.

**Contamination table:** for each model and dataset, record whether the model's paper or card reports training on that dataset or split. Analyse contaminated and uncontaminated pairs separately. Contamination itself is a confound worth reporting: a model that memorised a dataset's reporting style may look image-independent for the wrong reason.

---

## 6. Phase 2: Preregister Study 2 (2–3 days)

**Written and committed before any Study 2 generation.**

**Sample size**, from the Study 1 effects:

- The between-model difference in source-following was about +0.28 on the judge (per-study SD about 1.6) and +0.12 on CheXbert (SD about 0.6).
- 80% power at α = 0.05 needs about **250 pairs** for the judge and about **200** for CheXbert.
- Target: **about 300 contrast pairs per dataset**, or the maximum the dataset supports (MIMIC's test split may cap lower), plus 50 same-label pairs.

**Primary outcome:** the difference in source-following between the specialist and generalist **model classes**, on contrast pairs, measured with the primary scorer chosen in Phase 0. It's pooled across datasets in a mixed-effects model, with dataset and study as random effects.

**Key secondary outcomes:**

1. **Image-attributable agreement** (C1 minus the no-information baseline), per model and dataset.
2. **The base-rate test.** Generalists' image-free (C5) agreement should be high on abnormal-heavy datasets (MIMIC) and low on normal-heavy ones (IU X-ray). This is the mechanistic prediction from Study 1, and the most novel claim available.
3. **The medical fine-tuning ablation:** Lingshu vs Qwen2.5-VL.
4. **Finding tiers:** image dependence by how guessable a finding is.
5. **Scorer dependence:** whether conclusions change between holistic and finding-level scorers. The Study 1 result becomes a planned test.

**Interpretation rules:** the positive control must pass per dataset, and scorers must agree, as in Study 1.

---

## 7. Phase 3: Run (about 3–5 weeks of queue time)

**Rough volume:** 8 models × 3 datasets × about 350 studies × 5 conditions ≈ **42,000 generations**.

**GPU time:** at roughly 2–6 seconds per generation, that's about **25–70 GPU-hours**, plus scoring:

- the judge and GREEN at 7 items per study ≈ 59,000 items each,
- RadGraph and CheXbert on CPU.

With one slice and queue waits, expect several weeks of runs. Ordering:

1. **Datasets in order of readiness:** IU X-ray first (public and small, so it shakes out the pipeline), then MIMIC, then CheXpert Plus.
2. **Within each dataset:** the positive control (MAIRA-2) first, then the generalists, then the ablation pair.
3. **After each dataset completes,** run the scorers, then check the positive control before starting the next dataset. Stop and diagnose if it fails.

**Robustness:** all runners resume after interruption. Every run is tagged with model version, precision, decoding settings and git commit.

---

## 8. Phase 4: Analysis and write-up (about 2–3 weeks)

- **Mixed-effects models:** agreement modelled by condition × model class × dataset, with random effects for study and model. This answers whether specialist vs generalist differences hold across datasets.
- **Per-model, per-dataset effect tables,** carried forward from Study 1.
- **The base-rate figure:** each model's image-free agreement plotted against each dataset's abnormality prevalence.
- **Tier analysis** from cached CheXbert and RadGraph labels.
- **Study 1 reported unchanged** as the pilot and preregistered exploratory study. Study 2 is the confirmatory study.

---

## 9. Timeline (indicative, solo, part-time)

| Weeks | Work |
| --- | --- |
| 1 | Phase 0 (RadGraph, GREEN, ReXVal validation); finish MedGemma for Study 1; request CheXpert Plus access |
| 2–3 | Model harness and adapters; smoke-test candidate models; IU X-ray adapter; contamination table |
| 3 | Study 2 preregistration committed |
| 4–8 | Generation runs (IU X-ray, then MIMIC, then CheXpert Plus) |
| 6–9 | Scoring and analysis, dataset by dataset |
| 9–11 | Write-up |

**Checkpoints with your professor:**

- after Phase 0, to agree on the primary scorer,
- after the IU X-ray positive control, to confirm the method generalizes,
- before the write-up, to choose a venue.

---

## 10. Risks and fallbacks

| Risk | Fallback |
| --- | --- |
| CheXpert Plus access is slow | Use IU X-ray plus MIMIC (two populations still test the base-rate mechanism); add PadChest or CheXpert later |
| A candidate model won't run (format, licence, memory) | Drop it; keep at least two models per role |
| MIMIC's test split can't supply 300 targets | Use all eligible test patients, and rely on pooling across datasets for power |
| ReXVal shows all scorers track radiologists poorly | Report that as a finding; use the scorer ensemble, and narrow the claims |
| The positive control fails on a dataset | Stop for that dataset and diagnose. Possible causes: image format, adapter, or a genuine population effect |
| Someone else publishes the report-generation audit first | Post Study 1 as a preprint early; Study 2's multi-dataset base-rate test and scorer validation remain distinct contributions |

---

## 11. Immediate next steps

1. Add the deviation note to the Study 1 preregistration (no human verification), and commit it.
2. Finish MedGemma for Study 1: generation, judge, CheXbert, analysis.
3. Set up RadGraph F1 on CPU, then GREEN on the GPU.
4. Request ReXVal access on PhysioNet if your credential doesn't already cover it, and CheXpert Plus access from Stanford AIMI.
5. Start the shortlist check: availability, licence and input format for each candidate model.