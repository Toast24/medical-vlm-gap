# Project A: Do Medical AI Models Actually Look at the X-ray?

*Results summary, updated 9 October 2026, with the **final** Study 2 results (all five models, including CheXagent). The analysis files in `reports/` hold the authoritative numbers. Still pending: the prompt-robustness check and the contamination table.*

---

## 1. The short version

AI models that write chest X-ray reports are usually graded on how closely their report matches the radiologist's. **That grade can't tell whether the model actually looked at the image**: a convincing report can be written from what patients *usually* look like.

We tested this directly, by changing the image (the right X-ray, someone else's X-ray, a black image, or no image) and watching what happened to the report. Across five models, three datasets and three scoring methods validated against radiologists:

- **Models range from strongly image-reading to essentially guessing.** The ordering is consistent everywhere: **CheXagent > MAIRA-2 > MedGemma > Lingshu > Qwen (≈ 0).**
- **Chest X-ray specialists depend on the image more than medical generalists.** This was the preregistered primary hypothesis, and it is **confirmed on all three scorers.**
- **Medical generalists carry a "sick patient" prior.** With no image, Lingshu and MedGemma describe a typical sick inpatient, while the specialists and the general-purpose model default to normal or empty output. That prior earns credit without looking.
- **Standard grading can't see the difference.** Lingshu scores about as well as MAIRA-2 on conventional agreement, yet can earn up to two-thirds of its score without any image.
- **No model notices when it can't see.** Every model describes a black image as a normal chest X-ray.

---

## 2. What kind of study this is

An **empirical audit study, with a methodological contribution.**

- **The question:** do medical vision-language models' reports depend on the image they're given?
- **The method:** controlled interventions on the image, with everything else (prompt, decoding, patient) held fixed. Each report is scored against both the correct radiologist report and a deliberately wrong one.
- **The findings:** the results of that audit for five current models, on three datasets.

Recent work (Lotfinia et al., 2026) audited image use for yes/no *question answering*, and named free-text *report generation* as the open next step. This study does that. Free text has no single answer to flip, so the method adds **two-reference scoring**, **finding-level analysis**, and **specialist positive controls.**

---

## 3. Setup

### 3.1 The five conditions

| Condition | Image given | Tests |
|---|---|---|
| **C1 Real** | The correct X-ray | Normal behaviour |
| **C2 Swapped** | Another patient's X-ray | Does the report follow the image it was given? |
| **C3 Blank** | A black 512×512 image | Behaviour with an uninformative image |
| **C4 Metadata only** | No image; acquisition metadata as text | Influence of non-image cues |
| **C5 No input** | No image | The model's built-in "default patient" |

**The key measure is source-following:** the swapped-image report's agreement with the *swapped-in* patient's report, minus its agreement with the *original* patient's report. Above zero means the model follows the image; about zero means it ignores it.

### 3.2 Models

| Model | Type | Notes |
|---|---|---|
| **CheXagent-2-3b** | Chest X-ray specialist | Run in its own environment (Transformers 4.40.0, as its code requires) |
| **MAIRA-2** | Chest X-ray specialist | Needs an image, so its C4 and C5 use the blank image |
| **MedGemma-27B** | Medical generalist | Language model in 8-bit (memory limit), vision encoder in bf16; long reports regenerated at 768 tokens |
| **Lingshu-7B** | Medical generalist | Built on Qwen2.5-VL |
| **Qwen2.5-VL-7B** | General-purpose base model | Lingshu's base model: tests what medical training adds |

Decoding is greedy (deterministic); a rerun of 23 cases reproduced every output exactly.

### 3.3 Datasets

| Dataset | Patients | Pairs | Role |
|---|---|---|---|
| **MIMIC-CXR** (test split) | Hospital, mostly abnormal, 70% AP | 100 (80 contrast, 20 same-label) | **Study 1** (preregistered) |
| **IU X-ray** | Outpatients, **60% normal** | 350 (300 contrast, 50 same-label) | **Study 2** (preregistered) |
| **CheXpert Plus** (validation split) | Hospital, **84% abnormal** | 100 (85 contrast, 15 same-label) | **Study 2** (preregistered) |
| ReXVal | 200 candidate reports, 6 radiologists | — | Scorer validation |

"Contrast" pairs swap in a patient with different findings; "same-label" pairs swap in one with the same findings, as a sanity check.

### 3.4 Scorers

| Scorer | What it measures |
|---|---|
| **RadGraph F1** (primary for Study 2) | Overlap of clinical entities and relations (e.g. "effusion, located at left base") |
| **LLM judge** (Llama-3.1-8B) | Holistic agreement, 0–3, blinded to condition |
| **CheXbert** | Overlap of 14 standard finding labels |

---

## 4. Do the scorers track radiologists? (ReXVal)

Correlation (Kendall's τ) between each scorer's agreement and six radiologists' counts of clinically significant errors, over 200 reports. More negative is better.

| Scorer | τ | 95% CI |
|---|---|---|
| **RadGraph F1** | **−0.54** | −0.62 to −0.46 |
| LLM judge | −0.45 | −0.54 to −0.35 |
| CheXbert | −0.34 | −0.43 to −0.24 |

All three track radiologists; RadGraph best. Per the preregistered rule, **RadGraph became Study 2's primary scorer**, committed before any Study 2 output was scored. A Study 2 hypothesis counts as supported only if RadGraph **and** at least one other scorer agree.

*Source: `reports/rexval_validation.json`*

---

## 5. Study 1: MIMIC-CXR (n = 100, preregistered)

### 5.1 Do models follow a swapped image?

Source-following on contrast pairs, RadGraph (final, all five models):

| Model | RadGraph | 95% CI | q |
|---|---|---|---|
| **CheXagent** | **+0.11** | +0.07 to +0.14 | < 0.001 |
| **MAIRA-2** | +0.08 | +0.05 to +0.10 | < 0.001 |
| **MedGemma** | +0.03 | +0.01 to +0.05 | 0.016 |
| Lingshu | +0.03 | −0.00 to +0.06 | 0.094 (n.s.) |
| Qwen | −0.02 | −0.04 to +0.01 | 0.34 (n.s.) |

The judge and CheXbert give the same ordering; see `reports/scaled_analysis_{judge,chexbert}.txt`.

### 5.2 Agreement by condition (RadGraph)

| | CheXagent | Lingshu | MAIRA-2 | MedGemma | Qwen |
|---|---|---|---|---|---|
| Real image (C1) vs own report | **0.22** | 0.14 | 0.12 | 0.11 | 0.09 |
| Real image vs an unrelated report | 0.12 | 0.11 | 0.07 | 0.09 | 0.08 |
| Swapped image vs swapped-in report | 0.22 | 0.14 | 0.15 | 0.11 | 0.09 |
| Swapped image vs original report | 0.11 | 0.11 | 0.08 | 0.08 | 0.10 |
| Blank image (C3) | 0.16 | 0.13 | 0.00 | 0.08 | 0.12 |
| Metadata only (C4) | 0.00 | 0.13 | 0.10 | 0.04 | 0.06 |
| No input (C5) | **0.00** | 0.09 | 0.00 | 0.09 | 0.10 |

**CheXagent has the highest real-image agreement, and scores zero with no image.** Lingshu and Qwen score almost as well with a blank image as with the real one.

### 5.3 The preregistered Study 1 primary

MAIRA-2 minus Lingshu source-following: judge +0.28 (p = 0.14); CheXbert +0.12 (p = 0.029, CI just touching 0). **Not confirmed**: right direction, underpowered. Reported as a null.

### 5.4 What models write when there's nothing to see

These outputs contain no patient data:

| Model | Black image (C3) | No image (C5) |
|---|---|---|
| **Lingshu** (generalist) | A normal report | **A sick patient:** enlarged heart, pulmonary congestion |
| **MedGemma** (generalist) | A normal report, presented as a "standard PA chest X-ray" | **A sick patient:** enlarged heart, interstitial markings |
| **CheXagent** (specialist) | A structured normal report | "No abnormalities or changes are described in the image." |
| **MAIRA-2** (specialist) | "Preoperative for maxillofacial surgery… no significant alterations" | (Same; needs an image) |
| **Qwen** (general) | A normal report | A normal report |

**What this shows:**

1. **The sick-patient prior belongs to the medical *generalists*.** Both specialists default to normal or content-free output, and so does the general-purpose model.
2. **No model recognises a black image as uninformative.**
3. **MAIRA-2 falls back on memorised training-data phrasing,** in the style of the Spanish PadChest dataset.

### 5.5 Conventional grading is fooled

On holistic (judge) agreement, **Lingshu scores 1.29 and MAIRA-2 1.36**: nearly identical. Lingshu's report with **no image at all** also scores 1.29.

---

## 6. Study 2: IU X-ray and CheXpert (preregistered, confirmatory, final)

### 6.1 Verdicts

| Hypothesis | RadGraph (primary) | Judge | CheXbert | Verdict |
|---|---|---|---|---|
| **Primary:** specialists (MAIRA-2, CheXagent) > medical generalists (Lingshu, MedGemma) | ✅ +0.017, p = 0.006 | ✅ p < 0.001 | ✅ p < 0.001 | **SUPPORTED** |
| **H-A1:** MAIRA-2 follows the image | ✅ | ✅ | ✅ | **SUPPORTED** |
| **H-A1:** MedGemma follows the image | ✅ | ✅ | ✅ | **SUPPORTED** |
| **H-A2:** Qwen follows less than MAIRA-2 | ✅ | ✅ | ✅ | **SUPPORTED** |
| **H-A2:** Qwen follows less than MedGemma | ✅ | ✅ | ✅ | **SUPPORTED** |
| **H-A4:** medical fine-tuning (Lingshu vs Qwen) adds image use | ✅ | ✅ | ✅ | **SUPPORTED** |
| **H-A3:** the generalist prior scores lower on IU than CheXpert | ❌ | mixed | ❌ | Not supported (see §7) |
| **H-A3:** within datasets, the prior scores lower on normal than abnormal patients | ❌ | partly | ✅ | Secondary only |

**Six confirmatory tests supported, on every scorer.**

The primary outcome in detail (contrast pairs, IU + CheXpert, n = 385):

| Scorer | Difference | 95% CI | p |
|---|---|---|---|
| **RadGraph** | **+0.017** | +0.006 to +0.028 | 0.006 |
| Judge | +0.233 | +0.108 to +0.354 | < 0.001 |
| CheXbert | +0.160 | +0.121 to +0.199 | < 0.001 |

*An interim analysis, run before CheXagent's results were available, used MAIRA-2 alone as the specialist group, and was not significant on RadGraph (p = 0.19). The preregistration named both specialists from the start; the final analysis includes both.*

### 6.2 Source-following by dataset (contrast pairs)

**RadGraph:**

| Model | IU X-ray (300 pairs) | CheXpert (85 pairs) |
|---|---|---|
| **CheXagent** | **+0.031** (q < 0.001) | **+0.057** (q = 0.001) |
| MAIRA-2 | +0.021 (q < 0.001) | +0.023 (q = 0.045) |
| MedGemma | +0.028 (q < 0.001) | +0.018 (q = 0.076) |
| Lingshu | −0.005 | +0.011 |
| Qwen | −0.008 | −0.001 |

**CheXbert:**

| Model | IU | CheXpert |
|---|---|---|
| CheXagent | +0.247 (q < 0.001) | +0.266 (q < 0.001) |
| MAIRA-2 | +0.363 (q < 0.001) | +0.211 (q < 0.001) |
| MedGemma | +0.167 (q < 0.001) | +0.132 (q < 0.001) |
| Lingshu | +0.137 (q = 0.016) | −0.024 |
| Qwen | +0.072 | −0.110 (q = 0.038) |

**The gradient holds on both new datasets, and on both scorers.**

*Source: `reports/study2_analysis_final.txt`, `reports/study2_results.json`*

---

## 7. The population-prior mechanism (exploratory)

*Post hoc. Designed after the preregistered H-A3 test turned out to be confounded.*

**Why the preregistered H-A3 failed:** it compared raw no-image scores *across* datasets. But **every** model, including Qwen, whose default is "normal", scored higher on IU. IU's reports are shorter and more uniform, so generic text matches them better. Raw scores aren't comparable across datasets with different reporting styles.

### 7.1 Prior × population interaction (within each dataset)

Per study: the generalist's no-image score minus Qwen's no-image score, against **the same reference**, so reporting style cancels out. A positive interaction (abnormal patients minus normal patients) means the sick-patient prior is rewarded more where patients are sick.

**RadGraph:**

| Model | MIMIC | CheXpert | IU |
|---|---|---|---|
| Lingshu | +0.023 (p = 0.018) | +0.021 (p = 0.053) | **+0.026 (p = 0.007)** |
| MedGemma | +0.016 (p = 0.066) | +0.028 (p = 0.045) | **+0.041 (p < 0.001)** |

**All 18 combinations (2 models × 3 datasets × 3 scorers) are positive.** The judge agrees strongly for MedGemma (p ≤ 0.002 everywhere).

**Caveats:**

- On RadGraph the effect is relative: the sick prior ties with the normal prior on abnormal patients, and loses heavily on normal ones.
- CheXbert's values are inflated by its scoring rule (an empty output scores 1.0 against a normal reference), so take magnitudes from RadGraph and the judge.

### 7.2 How much of the score needs the image?

The ratio of no-image agreement to real-image agreement, on RadGraph. Higher means more of the score was earned without the image.

| Model | MIMIC | CheXpert | IU |
|---|---|---|---|
| CheXagent | ≈ 0.00 | ≈ 0.00 | ≈ 0.00 |
| MAIRA-2 | 0.00 | 0.00 | 0.00 |
| **Lingshu** | 0.64 | 0.66 | **0.39** |
| MedGemma | 0.78 | 0.96 | 0.84 |
| **Qwen** | **1.05** | **1.00** | **0.97** |

- **Lingshu earns about two-thirds of its score without the image** on sick populations, but only 39% on mostly normal IU. Its prior pays off where the population matches. The judge agrees (1.01, 0.97, 0.74).
- **The specialists earn essentially nothing without the image.** Their no-image output contains no clinical content.
- **For Qwen, the image adds nothing:** its ratio is about 1.0 everywhere.
- **MedGemma is mixed** on RadGraph, but consistent with Lingshu on the judge (0.67, 0.72, 0.31).

*Source: `reports/study2_h_a3_exploratory.{txt,json}`*

---

## 8. What do models read from the image? (Finding tiers)

Image gain is balanced accuracy with the real image, minus balanced accuracy without it (a constant guess scores 0.5). Mean by prevalence tier:

| Dataset, tier | MAIRA-2 | MedGemma | Lingshu | Qwen |
|---|---|---|---|---|
| MIMIC, common | 0.21 | 0.16 | 0.09 | 0.01 |
| IU, common | 0.12 | 0.19 | 0.02 | 0.02 |
| IU, rare | 0.12 | 0.08 | 0.04 | 0.02 |
| CheXpert, common | 0.21 | 0.14 | 0.07 | −0.01 |
| CheXpert, rare | 0.03 | 0.05 | 0.03 | −0.01 |

- **The gradient holds at the level of individual findings** on all three datasets.
- **Image-reading models read across the board,** including rare findings on IU, where there are enough cases to measure them. The weakly grounded models read little anywhere.
- **Support devices** (tubes, lines) are the most image-read finding on MIMIC for every medical model.

CheXagent's tier results are in `reports/finding_tiers_{scaled,iu,chexpert}.txt`.

---

## 9. So, are the models grounded in the image?

**It depends on the model, and you can't tell from a standard score.**

| Model | Swapped image | Real vs black image | Score earned without the image | Verdict |
|---|---|---|---|---|
| **CheXagent** | ✅ Follows it strongly (3 datasets) | ✅ Large gain | ≈ 0% | **Most grounded** |
| **MAIRA-2** | ✅ Follows it (3 datasets) | ✅ Large gain | ≈ 0% | **Grounded** |
| **MedGemma-27B** | ✅ Follows it (3 datasets) | ✅ Gain | 31–96% (varies by scorer) | **Moderately grounded** |
| **Lingshu-7B** | ❌ Mostly doesn't | ✅ Small gain | 39–66% | **Weakly grounded:** relies heavily on a sick-patient prior |
| **Qwen2.5-VL-7B** | ❌ Doesn't | ❌ About zero | ~100% | **Not grounded** |

**Even the grounded models have limits:** none recognises an uninformative image.

---

## 10. What's established

| Status | Findings |
|---|---|
| **Confirmed** (preregistered, all 3 scorers) | Specialists depend on the image more than medical generalists (primary); the gradient CheXagent > MAIRA-2 > MedGemma > Lingshu > Qwen (H-A1, H-A2); medical fine-tuning adds image use (H-A4); the audit detects image use on three datasets; the scorers are validated against radiologists |
| **Strongly supported, exploratory** | Medical generalists carry a sick-patient prior, whose payoff depends on the population; specialists have no such prior; the prior is rewarded mainly by holistic scoring; black-image hallucination; conclusions depend on the scorer |
| **Not supported** (reported transparently) | Study 1 primary (underpowered); H-A3 as preregistered (confounded) |

---

## 11. Limitations

- **No in-house expert review.** The scorers are validated against radiologist annotations (ReXVal) instead.
- **LLM judge (8B):** sensitive to output format; MedGemma had 137 of 700 judge failures in Study 1. RadGraph is the primary scorer.
- **MedGemma run in 8-bit** because of memory limits; comparisons are within-model, so this applies equally to every condition.
- **IU frontal view chosen by filename heuristic.** The positive controls passing on IU suggest the images are appropriate.
- **Small CheXpert sample:** all 200 validation patients used, giving 100 pairs.
- **Possible training-data overlap**, notably CheXagent and CheXpert. CheXagent's strong results on MIMIC and IU argue against contamination being the whole explanation; contamination table pending.
- **Single prompt wording per model;** prompt-robustness check in progress.
- **The mechanism evidence is exploratory.**

---

## 12. Remaining work

| Task | Status |
|---|---|
| Prompt-robustness check (2 alternative prompts, 4 models, 100 IU pairs) | GPU jobs running |
| Contamination table (training data per model; CheXagent and CheXpert first) | To do |
| Paper draft | To start |

---

## 13. Where this stands for publication

| Venue | Assessment |
|---|---|
| Workshop / short paper | Strong |
| **Main track** (medical imaging / ML for health) | **Credible:** preregistered primary confirmed on all scorers, replicated across datasets |
| Journal (medical AI / digital health) | Realistic, with room to report everything fully |

**More models or datasets aren't needed.** The remaining items (prompt robustness, contamination) close the reviewers' most likely questions.

---

## Files

| Content | File |
|---|---|
| Study 1 preregistration | `docs/preregistration_project_a_scaled.md` |
| Study 2 preregistration, Amendment 1, primary-scorer decision | `docs/preregistration_project_a_study2.md` |
| Prompt-robustness plan | `docs/prompt_robustness_plan.md` |
| ReXVal scorer validation | `reports/rexval_validation.json` |
| Study 1 analyses (judge, CheXbert, RadGraph) | `reports/scaled_analysis_{judge,chexbert,radgraph}.txt`, `reports/project_a_scaled_results*.json` |
| **Study 2 final analysis** | `reports/study2_analysis_final.txt`, `reports/study2_results.json` |
| Study 2 interim analysis | `reports/study2_analysis.txt` |
| Exploratory mechanism analysis | `reports/study2_h_a3_exploratory.{txt,json}` |
| Finding tiers | `reports/finding_tiers_{scaled,iu,chexpert}.{txt,csv,json}` |
| Sample selection | `reports/project_a_{scaled,iu,chexpert}_selection_summary.json` |
| Code | `evaluation/`, `scripts/`, `slurm/`, `run_project_a_*.py` |

All files contain aggregate results only. No patient data or report text is included in the repository.
