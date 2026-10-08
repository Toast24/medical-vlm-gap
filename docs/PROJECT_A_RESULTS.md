# Project A: Do Medical AI Models Actually Look at the X-ray?

*Results summary, compiled 8 October 2026. The analysis files in `reports/` are the authoritative numbers. This document collates them and explains what they mean. Study 2's primary outcome is **interim** until CheXagent's scoring is complete.*

---

## 1. The short version

When AI models write chest X-ray reports, they are usually graded on how closely their report matches the radiologist's. **That grade cannot tell whether the model actually looked at the image.** A model can write a convincing report by describing what patients *usually* look like.

We tested this directly. We changed the image (the right X-ray, someone else's X-ray, a black image, or no image) and watched what happened to the report. Across five models, three datasets and three independently validated scoring methods:

- **Models range from genuinely image-reading to essentially guessing.** MAIRA-2 and MedGemma clearly read the image. Lingshu does so only weakly. Qwen2.5-VL, a general-purpose model, effectively doesn't.
- **On standard grading, a weakly grounded model can look as good as a strongly grounded one.** Lingshu scores about as well as MAIRA-2, but earns up to two-thirds of its score without any image.
- **Medical training gives models a "typical patient" prior.** With no image, the medical models describe a sick hospital patient; the general model describes a normal chest. How much that guess is rewarded depends on the patient population being evaluated.
- **No model notices when it can't see.** Every model describes a black image as a normal chest X-ray.

---

## 2. What kind of study this is

An **empirical audit study, with a methodological contribution.**

- **The question:** do medical vision-language models' reports depend on the image they're given?
- **The method:** controlled interventions on the image. Everything else (the prompt, decoding, the patient) stays fixed, while the image is changed. Each report is scored against both the correct radiologist report and a deliberately wrong one.
- **The findings:** the results of applying that audit to five current models on three datasets.

Recent work (Lotfinia et al., 2026) audited image use for yes/no *question answering*, and named free-text *report generation* as the open next step. This study does that. Free text has no single answer to flip, so the method adds **two-reference scoring**, **finding-level analysis**, and a **specialist positive control**.

---

## 3. Setup

### 3.1 The five conditions

Every study is run through all five, with the prompt and decoding fixed:

| Condition | Image given | Tests |
|---|---|---|
| **C1 Real** | The correct X-ray | Normal behaviour |
| **C2 Swapped** | Another patient's X-ray | Does the report follow the image it was given? |
| **C3 Blank** | A black 512×512 image | Behaviour with an uninformative image |
| **C4 Metadata only** | No image; acquisition metadata as text | Influence of non-image cues |
| **C5 No input** | No image | The model's built-in "default patient" |

**The key measure is source-following:** the swapped-image report's agreement with the *swapped-in* patient's report, minus its agreement with the *original* patient's report. A model that reads the image scores clearly above zero. A model that ignores it scores about zero.

### 3.2 Models

| Model | Type | Notes |
|---|---|---|
| **MAIRA-2** | Chest X-ray specialist | Needs an image, so its C4 and C5 use the blank image |
| **CheXagent-2-3b** | Chest X-ray specialist | Results still being completed |
| **MedGemma-27B** | Medical generalist | Language model in 8-bit (memory limit); vision encoder in bf16 |
| **Lingshu-7B** | Medical generalist | Built on Qwen2.5-VL |
| **Qwen2.5-VL-7B** | General-purpose base model | Lingshu's base model: a clean test of what medical training adds |

Decoding is greedy (deterministic); a rerun of 23 cases reproduced every output exactly.

### 3.3 Datasets

| Dataset | Patients | Pairs | Role |
|---|---|---|---|
| **MIMIC-CXR** (test split) | Hospital, mostly abnormal, 70% AP | 100 (80 contrast, 20 same-label) | **Study 1** (preregistered) |
| **IU X-ray** | Outpatients, **60% normal** | 350 (300 contrast, 50 same-label) | **Study 2** (preregistered) |
| **CheXpert Plus** (validation split) | Hospital, **84% abnormal** | 100 (85 contrast, 15 same-label) | **Study 2** (preregistered) |
| ReXVal | 200 candidate reports, 6 radiologists | — | Scorer validation |

"Contrast" pairs swap in a patient with different findings, so following the image is detectable. "Same-label" pairs swap in a patient with the same findings, as a sanity check.

### 3.4 Scorers

| Scorer | What it measures |
|---|---|
| **RadGraph F1** | Overlap of clinical entities and their relations (e.g. "effusion, located at left base") |
| **LLM judge** (Llama-3.1-8B) | Holistic agreement, 0–3, blinded to condition |
| **CheXbert** | Overlap of 14 standard finding labels |

---

## 4. Do the scorers track radiologists? (ReXVal validation)

Each scorer was checked against six radiologists' counts of clinically significant errors in 200 candidate reports. A good scorer gives a strongly **negative** correlation (Kendall's τ): more agreement should mean fewer errors.

| Scorer | τ vs clinically significant errors | 95% CI |
|---|---|---|
| **RadGraph F1** | **−0.54** | −0.62 to −0.46 |
| LLM judge | −0.45 | −0.54 to −0.35 |
| CheXbert | −0.34 | −0.43 to −0.24 |

**Interpretation:** all three track radiologists, and RadGraph does best. Under the preregistered rule, **RadGraph became Study 2's primary scorer**. This decision was committed before any Study 2 output was scored. A Study 2 hypothesis counts as supported only if RadGraph **and** at least one other scorer agree.

*Source: `reports/rexval_validation.json`*

---

## 5. Study 1: MIMIC-CXR (n = 100, preregistered)

### 5.1 Do models follow a swapped image?

Source-following on contrast pairs:

| Model | Judge (0–3) | CheXbert (0–1) | RadGraph (0–1) | Verdict |
|---|---|---|---|---|
| **MAIRA-2** | +0.49 (q = 0.009) | +0.23 (q < 0.001) | +0.08 (q < 0.001) | ✅ Follows the image |
| **MedGemma-27B** | +0.54 (q = 0.006) | +0.18 (q < 0.001) | +0.03 (q = 0.018) | ✅ Follows the image |
| **Lingshu** | +0.16 (n.s.) | +0.11 (q = 0.056) | +0.03 (q = 0.11) | ⚠️ Weak at best |
| **Qwen2.5-VL** | −0.29 (n.s.) | +0.01 (n.s.) | −0.02 (n.s.) | ❌ Does not |

**The same ordering appears on every scorer: MAIRA-2 > MedGemma ≥ Lingshu > Qwen.** The scales differ, so compare signs and significance across scorers, not raw sizes. MedGemma's values were computed before its truncated reports were regenerated; the refreshed files in `reports/` are authoritative.

### 5.2 The preregistered primary outcome

MAIRA-2 minus Lingshu source-following, on contrast pairs:

| Scorer | Difference | 95% CI | p |
|---|---|---|---|
| Judge (preregistered primary) | +0.28 | −0.10 to +0.66 | 0.14 |
| CheXbert | +0.12 | −0.01 to +0.26 | 0.029 |

**Not confirmed.** It's in the right direction, but the study was powered for the pilot's effect size, which turned out about twice as large. Reported as a null.

### 5.3 What the models write when there's nothing to see

These outputs were produced without any patient data, so they can be shown directly:

| Model | Black image (C3) | No image (C5) |
|---|---|---|
| Lingshu | A normal report | **A sick-patient report:** enlarged heart, pulmonary congestion |
| MedGemma | A normal report, presented as a "standard PA chest X-ray" | **A sick-patient report:** enlarged heart, interstitial markings |
| Qwen2.5-VL | A normal report | A normal report |
| MAIRA-2 | "Preoperative for maxillofacial surgery… no significant alterations" | (Same; MAIRA-2 needs an image) |

**What this shows:**

1. **The medical models' default patient is a sick inpatient,** typical of their training data. The general-purpose model defaults to "normal". The prior comes from medical training.
2. **No model recognises a black image as uninformative.** All of them describe normal lungs.
3. **MAIRA-2 falls back on memorised training-data phrasing,** in the style of the Spanish PadChest dataset.

### 5.4 What do models actually read from the image? (Finding tiers)

Image gain is measured as balanced accuracy with the real image, minus balanced accuracy without it. A constant guess scores 0.5, so this measures improvement over a fixed guess.

| | MAIRA-2 | MedGemma | Lingshu | Qwen |
|---|---|---|---|---|
| **Common findings (≥25% prevalence), mean gain** | **0.21** | **0.16** | 0.09 | 0.01 |
| Support devices (tubes, lines) | 0.23 | 0.21 | 0.19 | 0.05 |
| Rare findings (<10%), mean gain | −0.01 | 0.16 | 0.00 | 0.04 |

**Interpretation:** image use concentrates on **conspicuous findings**, devices most of all, then effusion and cardiomegaly. It follows the same gradient as source-following. Rare findings show almost no image use (MedGemma's rare-tier value rests on very few cases). Our original prediction, that image use would concentrate on rare findings, was **not supported**.

*Source: `reports/finding_tiers_scaled.json`*

### 5.5 Conventional grading is fooled

On holistic (judge) agreement with the correct report, **Lingshu scores 1.29 and MAIRA-2 1.36**: nearly identical. But Lingshu's report with **no image at all** also scores 1.29. Standard grading can't tell that one model is reading the image and the other largely isn't.

---

## 6. Study 2: IU X-ray and CheXpert (preregistered, confirmatory)

### 6.1 Verdicts

| Hypothesis | RadGraph (primary) | Judge | CheXbert | Verdict |
|---|---|---|---|---|
| **H-A1:** MAIRA-2 follows the image | ✅ p < 0.001 | ✅ | ✅ | **SUPPORTED** |
| **H-A1:** MedGemma follows the image | ✅ p < 0.001 | ✅ | ✅ | **SUPPORTED** |
| **H-A2:** Qwen follows less than MAIRA-2 | ✅ p < 0.001 | ✅ | ✅ | **SUPPORTED** |
| **H-A2:** Qwen follows less than MedGemma | ✅ p < 0.001 | ✅ | ✅ | **SUPPORTED** |
| **H-A4:** Lingshu gains more from the image than Qwen (medical fine-tuning) | ✅ p < 0.001 | ✅ | ✅ | **SUPPORTED** |
| **H-A3:** the medical prior scores lower on IU than CheXpert | ❌ | mixed | ❌ | Not supported |
| **H-A3:** within each dataset, the prior scores lower on normal than abnormal targets | ❌ | partly | ✅ | Secondary only (not supported) |
| **Primary:** specialists > medical generalists | ❌ p = 0.19 | ✅ | ✅ | **Interim:** not supported on RadGraph |

**Five confirmatory tests passed on every scorer.** The gradient, and the effect of medical fine-tuning, replicate on new populations and a new institution.

### 6.2 Source-following by dataset (RadGraph, contrast pairs)

| Model | IU X-ray (300 pairs) | CheXpert (85 pairs) |
|---|---|---|
| MAIRA-2 | **+0.021** (q < 0.001) | +0.023 (q = 0.05) |
| MedGemma | **+0.028** (q < 0.001) | +0.018 (q = 0.08) |
| Lingshu | −0.005 | +0.011 |
| Qwen | −0.008 | −0.001 |

On IU, MAIRA-2 and MedGemma clearly follow the image, while Lingshu and Qwen sit at zero. CheXpert shows the same pattern with less statistical power (85 pairs). CheXbert agrees, more strongly.

### 6.3 Why the primary outcome is weak

The preregistered primary compares "specialists" with "medical generalists". **MedGemma, a generalist, behaves like a specialist,** which dilutes the comparison. The variation is model-specific, not a clean split between categories. The final primary analysis will include CheXagent as a second specialist.

*Source: `reports/study2_analysis.txt`, `reports/study2_results.json`*

---

## 7. Exploratory: the population-prior mechanism

*Post hoc. Designed after the preregistered H-A3 test turned out to be confounded.*

**Why the preregistered H-A3 failed:** it compared raw no-image scores *across* datasets. But **every** model scored higher on IU, including Qwen, whose default is "normal". IU's reports are shorter and more uniform, so generic text matches them better. Raw scores aren't comparable across datasets with different reporting styles.

### 7.1 Prior × population interaction (within each dataset)

For each study, the medical model's no-image score minus Qwen's no-image score, against **the same reference**, so reporting style cancels out. A positive interaction (abnormal patients minus normal patients) means the sick-patient prior is rewarded more where patients are sick.

**RadGraph:**

| Model | MIMIC | CheXpert | IU |
|---|---|---|---|
| Lingshu | +0.023 (p = 0.018) | +0.021 (p = 0.053) | **+0.026 (p = 0.007)** |
| MedGemma | +0.016 (p = 0.066) | +0.028 (p = 0.045) | **+0.041 (p < 0.001)** |

**All 18 combinations (2 models × 3 datasets × 3 scorers) are positive.** The judge agrees strongly for MedGemma (p ≤ 0.002 everywhere).

**Caveats:**

- On RadGraph the effect is relative: the sick prior ties with the normal prior on abnormal patients, and loses heavily on normal ones.
- CheXbert's values (+0.5 to +1.2) are inflated by its scoring rule: an empty "normal" output scores 1.0 against a normal reference. Take magnitudes from RadGraph and the judge.

### 7.2 How much of the score needs the image?

The ratio of no-image agreement to real-image agreement, on RadGraph. Higher means more of the score was earned without the image.

| Model | MIMIC | CheXpert | IU |
|---|---|---|---|
| **Lingshu** | 0.64 | 0.66 | **0.39** |
| MedGemma | 0.78 | 0.96 | 0.84 |
| **Qwen** | **1.05** | **1.00** | **0.97** |
| MAIRA-2 | 0.00 | 0.00 | 0.00 |

- **Lingshu earns about two-thirds of its score without the image** on sick populations, but only 39% on mostly normal IU. Its prior pays off where the population matches it. The judge agrees (1.01, 0.97, 0.74).
- **Qwen's ratio is about 1.0 everywhere: the image adds nothing.**
- **MedGemma is mixed** on RadGraph, though consistent with Lingshu on the judge (0.67, 0.72, 0.31).
- **MAIRA-2 scores 0** because its blank-image template contains no clinical entities.

*Source: `reports/study2_h_a3_exploratory.json`*

---

## 8. So, are the models grounded in the image?

**It depends on the model, and you can't tell from a standard score.**

| Model | Swapped image | Real vs black image | Score earned without the image | Verdict |
|---|---|---|---|---|
| **MAIRA-2** | ✅ Follows it (3 datasets) | ✅ Large gain | ~0% | **Grounded** |
| **CheXagent** | Pending | Pending | 8–29% (early, judge only) | **Probably grounded** |
| **MedGemma-27B** | ✅ Follows it (3 datasets) | ✅ Gain | 31–96% (varies by scorer) | **Moderately grounded** |
| **Lingshu-7B** | ❌ Mostly doesn't | ✅ Small gain | 39–66% | **Weakly grounded:** much of its score comes from a "typical patient" guess |
| **Qwen2.5-VL-7B** | ❌ Doesn't | ❌ About zero | ~100% | **Not grounded** |

**Even the grounded models are limited:** they read mainly conspicuous findings (devices, large effusions), and none recognises an uninformative image.

---

## 9. What's established

| Status | Findings |
|---|---|
| **Confirmed** (preregistered, replicated, all 3 scorers) | The audit detects image use; models differ in a consistent gradient (H-A1, H-A2); medical fine-tuning adds image use (H-A4); the scorers are validated against radiologists |
| **Strongly supported, exploratory** | The population prior, and its population-dependent payoff; the prior is rewarded mainly by holistic scoring; what models read from images; black-image hallucination; conclusions depend on the scorer |
| **Not supported** (reported transparently) | Study 1 primary (underpowered); Study 2 primary (interim; the category split doesn't hold); H-A3 as preregistered (confounded); image use concentrated in rare findings |

---

## 10. Limitations

- **No in-house expert review.** The scorers are validated against radiologist annotations (ReXVal) instead.
- **LLM judge (8B):** sensitive to output format; MedGemma had 137 of 700 judge failures in Study 1. RadGraph is the primary scorer.
- **MedGemma run in 8-bit** because of memory limits. Comparisons are within-model, so this applies equally to all conditions.
- **IU frontal view chosen by filename heuristic.** The positive control passing on IU suggests the selected images are appropriate.
- **CheXpert sample is small:** all 200 validation patients used, giving 100 pairs.
- **Single prompt wording** per model; prompt robustness not yet tested.
- **Possible training-data overlap** between models and datasets; contamination table to be compiled.
- **The mechanism evidence is exploratory.**

---

## 11. Still to do

| Task | Status |
|---|---|
| CheXagent: finish scoring, then the final Study 2 primary analysis | In progress |
| Finding tiers on IU and CheXpert | To do |
| Contamination table | To do |
| Prompt-robustness check (recommended for main-track submission) | Optional |
| Small radiologist review (if available) | Optional |
| Paper draft | To start |

---

## 12. Where this stands for publication

| Venue | Assessment |
|---|---|
| Workshop | Comfortably sufficient |
| Short paper / findings track | Strong |
| Main track (medical imaging / ML for health) | Competitive; strengthened by the prompt-robustness check and a small expert review |
| Journal (medical AI / digital health) | Realistic, with room to report everything fully |

**More models or datasets aren't needed.** Five models and three populations already cover the meaningful axes.

---

## Files

| Content | File |
|---|---|
| Study 1 preregistration | `docs/preregistration_project_a_scaled.md` |
| Study 2 preregistration, Amendment 1, primary-scorer decision | `docs/preregistration_project_a_study2.md` |
| ReXVal scorer validation | `reports/rexval_validation.json` |
| Study 1 analyses (judge, CheXbert, RadGraph) | `reports/scaled_analysis_{judge,chexbert,radgraph}.txt`, `reports/project_a_scaled_results*.json` |
| Study 1 finding tiers | `reports/finding_tiers_scaled.{csv,json,txt}` |
| Study 2 confirmatory analysis | `reports/study2_analysis.txt`, `reports/study2_results.json` |
| Exploratory H-A3 reanalysis | `reports/study2_h_a3_exploratory.{txt,json}` |
| Sample selection summaries | `reports/project_a_{scaled,iu,chexpert}_selection_summary.json` |
| Code | `evaluation/`, `scripts/`, `slurm/`, `run_project_a_*.py` |

All files contain aggregate results only. No patient data or report text is included in the repository.
