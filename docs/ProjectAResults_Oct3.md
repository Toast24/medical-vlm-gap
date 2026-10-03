# Project A: Results So Far

*Collated 3 October 2026. Covers the pilot (n = 20) and the preregistered scaled study (Study 1, n = 100). MedGemma-27B is in progress. All numbers are copied from the analysis outputs. Agreement scores: LLM judge 0–3; CheXbert label Jaccard 0–1.*

---

## 0. Summary

- **The method works.** Swapping and removing images, and scoring each output against both the correct reference and a wrong one, detects image dependence in a chest X-ray specialist (MAIRA-2) on both scorers.
- **The specialist depends on the image far more than the generalist.** On the like-for-like test (real image vs blank), MAIRA-2 gains much more than Lingshu-7B: q \< 0.001 on both scorers.
- **Lingshu uses the image, but weakly.** Its reports are study-specific, and on finding-level scoring the real image improves them slightly. On holistic scoring, its image-free report scores the same as its real-image reports.
- **Conclusions depend on the scorer.** Holistic and finding-level scoring disagree on whether Lingshu's image adds anything beyond its prior.
- **The preregistered primary outcome was not confirmed.** The MAIRA-2 vs Lingshu difference in source-following is in the predicted direction: not significant on the judge, borderline on CheXbert.

---

## 1. Setup

| Item | Pilot | Study 1 (scaled, preregistered) |
| --- | --- | --- |
| Dataset | MIMIC-CXR test split | MIMIC-CXR test split |
| Target studies | 20 (informal pairs) | 100 (70 abnormal, 30 normal); one per patient; pilot patients excluded |
| Pairs | Effectively random | 80 contrast (mean Jaccard 0.00) + 20 same-label (mean Jaccard 0.87); seed 2026 |
| Models | Lingshu-7B, MAIRA-2 | Lingshu-7B, MAIRA-2; MedGemma-27B (int8) in progress |
| Conditions | C1 real · C2 mismatched · C3 blank · C4 metadata only · C5 no input | Same |
| Decoding | Greedy (Lingshu: top_k = 1, repetition penalty 1.05) | Same |
| Scorers | LLM judge (Llama-3.1-8B, blinded) | LLM judge + CheXbert label agreement |
| Preregistration | — | `docs/preregistration_project_a_scaled.md` (commit 2fb6a27) |

**Model adaptations.** MAIRA-2 requires an image, so its C4 and C5 used the blank image (its C5 equals C3). Lingshu and MedGemma run C4 and C5 truly text-only.

**Scaled sample notes.**

- The MIMIC test split has about 300 patients, so the one-study-per-patient pool for targets was 240 (59 normal, 178 abnormal).
- Sources are drawn from 798 studies of non-target patients.
- Studies with uncertain CheXpert labels were excluded.
- About 70% of the pool is AP.

---

## 2. Pilot results (n = 20)

### 2.1 Lexical behaviour (Lingshu)

| Condition | Duplicate outputs (of 20) |
| --- | --- |
| C1 real | 0 |
| C2 mismatched | 1 |
| C3 blank | 19 |
| C4 metadata | 15 |
| C5 no input | 19 |

| Comparison | Mean token Jaccard |
| --- | --- |
| C1 vs C2 (same target) | 0.183 |
| C2(A←B) vs C1 of other studies | 0.215 |
| C1 vs C1 across studies ("house-style floor") | 0.216 |

C1 vs C2: TF-IDF cosine 0.434, ROUGE-L 0.213, 0/20 exact matches.

**Inference.** The image changes Lingshu's wording, and the no-image conditions collapse to near-identical text. But C1-vs-C2 overlap sits at the cross-study floor, so lexical difference says nothing about clinical grounding. As expected, no target information leaks into C2.

### 2.2 Reproducibility (Lingshu)

Regenerating 23 cases (20 C2, 3 C1) reproduced **23/23 outputs exactly** (Jaccard 1.000).

**Inference.** Generation is deterministic, so every difference between conditions is driven by the input. C2(A←B) is exactly Lingshu's output for image B.

### 2.3 Judge results (pilot)

Mean agreement by role:

| Role | Lingshu | MAIRA-2 |
| --- | --- | --- |
| C1 vs own | 1.70 | 1.47 |
| C1 vs unrelated | 1.26 | 0.63 |
| C2 vs source | 1.15 | 1.50 |
| C2 vs target | 1.10 | 0.80 |
| C3 vs own | 0.80 | 0.63 |
| C4 vs own | 0.70 | 0.50 |
| C5 vs own | 1.50 | 0.50 |

Paired effects:

| Effect | Lingshu | MAIRA-2 |
| --- | --- | --- |
| Source-following (C2 src − tgt) | +0.05 \[−0.45, +0.60\], p = 0.74 | +0.70 \[+0.10, +1.30\], p = 0.052 |
| Own vs unrelated (C1) | +0.42 \[−0.11, +0.95\], p = 0.16 | +0.78 \[+0.11, +1.33\], p = 0.035 |
| C1 − C3 (blank) | +0.90 \[+0.50, +1.30\], p = 0.003 | +0.89, p = 0.005 |
| C1 − C4 (metadata) | +1.00 \[+0.50, +1.50\], p = 0.004 | +0.95, p = 0.013 |
| C1 − C5 (no input) | +0.20 \[−0.30, +0.70\], p = 0.41 | +1.05, p = 0.002\* |

\*For MAIRA-2, C5 is the blank image.

Between-model (MAIRA-2 − Lingshu, paired bootstrap):

| Effect | Difference | 95% CI |
| --- | --- | --- |
| Source-following | +0.65 | \[−0.10, +1.45\] |
| Own vs unrelated | +0.47 | \[−0.47, +1.29\] |
| Real vs blank | −0.11 | \[−0.78, +0.56\] |
| Real vs no information (asymmetric) | +0.84 | \[+0.11, +1.58\] |

**Stratification.** 19/20 pilot targets were abnormal, so Lingshu's high C5 score couldn't come from matching normal studies.

**Pilot inference.** MAIRA-2 behaved like an image reader; Lingshu's image-free text scored nearly as well as its real-image reports. These findings motivated the preregistered scaled study.

---

## 3. Study 1: scaled, preregistered (n = 100)

### 3.1 Coverage

| Scorer | Lingshu | MAIRA-2 |
| --- | --- | --- |
| LLM judge | 683/700 items (17 parse failures) | 678/700 (22 failures) |
| CheXbert | 700/700 | 700/700 |

### 3.2 Primary outcome (preregistered: MAIRA-2 − Lingshu source-following, contrast pairs)

| Scorer | n | Mean | 95% CI | p |
| --- | --- | --- | --- | --- |
| **LLM judge (preregistered primary scorer)** | 71 | +0.28 | \[−0.10, +0.66\] | 0.138 |
| CheXbert (secondary scorer) | 80 | +0.12 | \[−0.01, +0.26\] | 0.029 (Wilcoxon) |

**Inference.** The effect is in the predicted direction, but **the preregistered primary test is not significant**. The effect is under half the pilot estimate (+0.65); the study was powered for 0.65, not 0.28. CheXbert is borderline: the Wilcoxon test is significant, but the bootstrap CI only just includes zero. Report this as directionally consistent, not confirmed.

### 3.3 Mean agreement by role

| Role | Judge: Lingshu | Judge: MAIRA-2 | CheXbert: Lingshu | CheXbert: MAIRA-2 |
| --- | --- | --- | --- | --- |
| C1 vs own | 1.29 | 1.36 | 0.24 | 0.38 |
| C1 vs unrelated | 0.86 | 0.57 | 0.13 | 0.18 |
| C2 vs source | 1.14 | 1.40 | 0.24 | 0.36 |
| C2 vs target | 1.03 | 0.97 | 0.15 | 0.16 |
| C3 vs own (blank) | 1.03 | 0.47 | 0.15 | 0.15 |
| C4 vs own (metadata) | 0.93 | 0.58 | 0.15 | 0.16 |
| C5 vs own (no input) | 1.29 | 0.58 | 0.13 | 0.15 |

### 3.4 Within-model effects: LLM judge (mean \[95% CI\], BH q)

**Source-following (C2 vs source − C2 vs target)**

| Stratum | Lingshu | MAIRA-2 |
| --- | --- | --- |
| All | +0.09 \[−0.20, +0.36\] q = 0.54 | +0.40 \[+0.11, +0.69\] q = 0.013 |
| **Contrast** | +0.16 \[−0.16, +0.48\] q = 0.34 | **+0.49 \[+0.18, +0.80\] q = 0.005** |
| Same-label | −0.21 \[−0.74, +0.32\] q = 0.54 | +0.05 \[−0.63, +0.74\] q = 1.0 |
| Normal | −0.14 \[−0.50, +0.21\] q = 0.54 | +0.24 \[−0.28, +0.76\] q = 0.33 |
| Abnormal | +0.18 \[−0.17, +0.55\] q = 0.34 | +0.47 \[+0.12, +0.81\] q = 0.026 |
| PA | −0.11 \[−0.56, +0.33\] q = 0.68 | +0.54 \[+0.06, +1.00\] q = 0.028 |
| AP | +0.21 \[−0.14, +0.55\] q = 0.27 | +0.31 \[−0.07, +0.67\] q = 0.17 |

**Own vs unrelated reference (C1)**

| Stratum | Lingshu | MAIRA-2 |
| --- | --- | --- |
| All | +0.47 \[+0.21, +0.72\] q = 0.002 | +0.81 \[+0.56, +1.06\] q \< 0.001 |
| Contrast | +0.60 \[+0.34, +0.87\] q \< 0.001 | +0.82 \[+0.55, +1.08\] q \< 0.001 |
| Same-label | −0.11 \[−0.67, +0.44\] q = 0.73 | +0.79 \[+0.10, +1.42\] q = 0.08 |
| Normal | +0.80 \[+0.40, +1.20\] q = 0.004 | +1.20 \[+0.87, +1.53\] q \< 0.001 |
| Abnormal | +0.31 \[+0.00, +0.62\] q = 0.08 | +0.63 \[+0.31, +0.95\] q = 0.002 |
| PA | +0.54 \[+0.16, +0.92\] q = 0.025 | +1.03 \[+0.59, +1.41\] q \< 0.001 |
| AP | +0.42 \[+0.10, +0.74\] q = 0.026 | +0.67 \[+0.36, +0.98\] q \< 0.001 |

**Real image vs blank (C1 − C3)**

| Stratum | Lingshu | MAIRA-2 |
| --- | --- | --- |
| All | +0.27 \[+0.06, +0.48\] q = 0.016 | +0.90 \[+0.67, +1.12\] q \< 0.001 |
| Contrast | +0.27 \[+0.04, +0.51\] q = 0.026 | +0.79 \[+0.53, +1.03\] q \< 0.001 |
| Same-label | +0.23 \[−0.23, +0.71\] q = 0.38 | +1.32 \[+0.90, +1.68\] q = 0.002 |
| Normal | +0.03 \[−0.30, +0.37\] q = 0.70 | +0.80 \[+0.33, +1.20\] q = 0.006 |
| Abnormal | +0.38 \[+0.12, +0.62\] q = 0.016 | +0.94 \[+0.68, +1.20\] q \< 0.001 |
| PA | +0.24 \[−0.11, +0.59\] q = 0.20 | +0.97 \[+0.59, +1.35\] q \< 0.001 |
| AP | +0.28 \[+0.02, +0.54\] q = 0.042 | +0.84 \[+0.57, +1.10\] q \< 0.001 |

**Real image vs no-information baseline** (Lingshu: C1 − C5; MAIRA-2: C1 − C3)

| Stratum | Lingshu | MAIRA-2 |
| --- | --- | --- |
| **All** | **−0.01 \[−0.25, +0.23\] q = 1.0** | +0.90 \[+0.67, +1.12\] q \< 0.001 |
| Contrast | +0.09 \[−0.17, +0.35\] q = 0.52 | +0.79 \[+0.53, +1.03\] q \< 0.001 |
| Same-label | −0.42 \[−0.84, +0.00\] q = 0.16 | +1.32 \[+0.90, +1.68\] q = 0.002 |
| Normal | +0.30 \[−0.03, +0.63\] q = 0.11 | +0.80 \[+0.33, +1.20\] q = 0.006 |
| Abnormal | −0.15 \[−0.46, +0.15\] q = 0.38 | +0.94 \[+0.68, +1.20\] q \< 0.001 |
| PA | +0.11 \[−0.27, +0.49\] q = 0.62 | +0.97 \[+0.59, +1.35\] q \< 0.001 |
| AP | −0.09 \[−0.39, +0.20\] q = 0.68 | +0.84 \[+0.57, +1.10\] q \< 0.001 |

### 3.5 Within-model effects: CheXbert (mean \[95% CI\], BH q)

**Source-following**

| Stratum | Lingshu | MAIRA-2 |
| --- | --- | --- |
| All | +0.10 \[+0.01, +0.18\] q = 0.027 | +0.20 \[+0.13, +0.27\] q \< 0.001 |
| **Contrast** | +0.11 \[+0.01, +0.21\] q = 0.048 | **+0.23 \[+0.15, +0.31\] q \< 0.001** |
| Same-label | +0.04 \[−0.02, +0.09\] q = 0.22 | +0.08 \[−0.04, +0.20\] q = 0.22 |
| Normal | +0.06 \[−0.17, +0.27\] q = 0.72 | +0.27 \[+0.11, +0.42\] q = 0.010 |
| Abnormal | +0.11 \[+0.04, +0.19\] q = 0.017 | +0.17 \[+0.10, +0.24\] q \< 0.001 |
| PA | +0.08 \[−0.06, +0.20\] q = 0.13 | +0.33 \[+0.23, +0.42\] q \< 0.001 |
| AP | +0.11 \[+0.00, +0.21\] q = 0.086 | +0.12 \[+0.03, +0.21\] q = 0.014 |

**Own vs unrelated reference (C1)**

| Stratum | Lingshu | MAIRA-2 |
| --- | --- | --- |
| All | +0.11 \[+0.04, +0.18\] q = 0.006 | +0.20 \[+0.12, +0.29\] q \< 0.001 |
| Contrast | +0.12 \[+0.03, +0.20\] q = 0.017 | +0.25 \[+0.15, +0.34\] q \< 0.001 |
| Same-label | +0.09 \[−0.01, +0.19\] q = 0.12 | +0.02 \[−0.14, +0.19\] q = 0.81 |
| Normal | +0.21 \[+0.03, +0.40\] q = 0.13 | +0.24 \[+0.06, +0.44\] q = 0.07 |
| Abnormal | +0.07 \[+0.01, +0.13\] q = 0.019 | +0.18 \[+0.10, +0.27\] q \< 0.001 |
| PA | +0.07 \[−0.04, +0.19\] q = 0.43 | +0.20 \[+0.05, +0.35\] q = 0.048 |
| AP | +0.14 \[+0.05, +0.23\] q = 0.006 | +0.21 \[+0.11, +0.31\] q \< 0.001 |

**Real image vs blank (C1 − C3)**

| Stratum | Lingshu | MAIRA-2 |
| --- | --- | --- |
| All | +0.09 \[+0.01, +0.16\] q = 0.007 | +0.23 \[+0.14, +0.32\] q \< 0.001 |
| Contrast | +0.05 \[−0.03, +0.13\] q = 0.038 | +0.20 \[+0.09, +0.30\] q = 0.001 |
| Same-label | +0.23 \[+0.03, +0.41\] q = 0.060 | +0.36 \[+0.25, +0.48\] q = 0.001 |
| Normal | −0.11 \[−0.28, +0.05\] q = 0.25 | −0.08 \[−0.28, +0.12\] q = 0.43 |
| Abnormal | +0.17 \[+0.11, +0.24\] q \< 0.001 | +0.36 \[+0.30, +0.43\] q \< 0.001 |
| PA | −0.03 \[−0.18, +0.11\] q = 1.0 | +0.10 \[−0.06, +0.27\] q = 0.22 |
| AP | +0.16 \[+0.09, +0.23\] q \< 0.001 | +0.31 \[+0.22, +0.39\] q \< 0.001 |

**Real image vs no-information baseline**

| Stratum | Lingshu | MAIRA-2 |
| --- | --- | --- |
| **All** | **+0.11 \[+0.04, +0.18\] q = 0.048** | +0.23 \[+0.14, +0.32\] q \< 0.001 |
| Contrast | +0.11 \[+0.04, +0.20\] q = 0.048 | +0.20 \[+0.09, +0.30\] q = 0.001 |
| Same-label | +0.09 \[−0.07, +0.28\] q = 0.37 | +0.36 \[+0.25, +0.48\] q = 0.001 |
| Normal | +0.29 \[+0.13, +0.45\] q = 0.015 | −0.08 \[−0.28, +0.12\] q = 0.43 |
| Abnormal | +0.03 \[−0.04, +0.10\] q = 0.68 | +0.36 \[+0.30, +0.43\] q \< 0.001 |
| PA | +0.08 \[−0.01, +0.19\] q = 0.31 | +0.10 \[−0.06, +0.27\] q = 0.22 |
| AP | +0.13 \[+0.03, +0.23\] q = 0.072 | +0.31 \[+0.22, +0.39\] q \< 0.001 |

*CheXbert caveat:* an output with no findings matches a normal reference perfectly (empty vs empty = 1.0), which flatters blank and empty outputs on normal studies. The abnormal stratum is the more meaningful CheXbert view.

### 3.6 Between-model (MAIRA-2 − Lingshu)

| Effect | Stratum | Judge: diff \[CI\] (q) | CheXbert: diff \[CI\] (q) |
| --- | --- | --- | --- |
| Source-following | All | +0.28 \[−0.07, +0.62\] (0.17) | +0.11 \[−0.00, +0.22\] (0.032) |
| Source-following | Contrast | +0.28 \[−0.10, +0.66\] (0.20) | +0.12 \[−0.01, +0.26\] (0.048) |
| Own vs unrelated | All | +0.36 \[+0.04, +0.68\] (0.042) | +0.09 \[+0.01, +0.17\] (0.025) |
| Own vs unrelated | Contrast | +0.24 \[−0.08, +0.57\] (0.22) | +0.13 \[+0.04, +0.21\] (0.001) |
| **Real vs blank** | All | **+0.65 \[+0.36, +0.95\] (\< 0.001)** | **+0.14 \[+0.07, +0.21\] (\< 0.001)** |
| **Real vs blank** | Contrast | **+0.54 \[+0.20, +0.88\] (0.007)** | **+0.14 \[+0.07, +0.21\] (\< 0.001)** |
| Real vs no info\* | All | +0.94 \[+0.59, +1.28\] (\< 0.001) | +0.12 \[−0.01, +0.24\] (0.032) |
| Real vs no info\* | Contrast | +0.70 \[+0.32, +1.08\] (0.004) | +0.09 \[−0.06, +0.22\] (0.12) |

\*Asymmetric: MAIRA-2's no-information baseline is the blank image, Lingshu's is true no-input.

### 3.7 Scorer agreement on the key questions

| Question | Judge | CheXbert | Agree? |
| --- | --- | --- | --- |
| Positive control: MAIRA-2 follows the swapped image (contrast) | +0.49, q = 0.005 | +0.23, q \< 0.001 | ✅ |
| Specialist gains more from the real image than the generalist (real vs blank) | +0.65, q \< 0.001 | +0.14, q \< 0.001 | ✅ |
| Lingshu's reports are study-specific (own vs unrelated) | +0.47, q = 0.002 | +0.11, q = 0.006 | ✅ |
| Lingshu follows the swapped image (contrast) | +0.16, n.s. | +0.11, q = 0.048 | ⚠️ |
| Lingshu's real image beats its image-free text (C1 − C5) | −0.01, n.s. | +0.11, q = 0.048 | ❌ |
| Primary: MAIRA-2 − Lingshu source-following | +0.28, p = 0.14 | +0.12, p = 0.029 (CI touches 0) | ⚠️ |

**Why they disagree on C1 − C5.** The judge scores reports holistically: Lingshu's image-free paragraph reads plausibly and earns partial credit (1.29), equal to its real-image reports. CheXbert scores only which findings are named: the image-free text names mostly wrong findings (0.13 vs 0.24 with the image). In short, the image improves *which findings Lingshu names*, slightly, but not the holistic impression of its reports.

### 3.8 Supporting observations

- **Same-label pairs behave as expected.** Source-following is near zero for both models on both scorers, since two studies with the same findings can't be distinguished.
- **View effect.** MAIRA-2's source-following is stronger on PA than AP on both scorers (judge +0.54 vs +0.31; CheXbert +0.33 vs +0.12), matching published question-answering audits.
- **Normal vs abnormal (judge).** Lingshu's C1 − C5 is +0.30 on normal studies and −0.15 on abnormal ones. This fits the population-prior mechanism: its image-free text resembles an abnormal hospital report.
- **No-input behaviour.** Lingshu's image-free outputs list several plausible findings; MAIRA-2's blank-image output is a short, near-empty report (about 13 words).

### 3.9 MedGemma-27B (in progress)

The smoke test passed:

- the vision encoder runs in bf16 and the language model in int8, using 29.3 GB of GPU memory;
- all five conditions succeeded, with images used for C1–C3 and none for C4–C5;
- C4 hit the 256-token limit, so the full run uses 384.

Full results are pending.

---

## 4. Inferences

1. **The intervention protocol is sensitive.** With the specialist as positive control, both scorers detect source-following and real-vs-blank gains. Null results for other models are therefore informative, not just a sign the measurement failed.
2. **The specialist is clearly more image-dependent than the generalist.** This is the most robust result between the models: the like-for-like real-vs-blank gain, at q \< 0.001 on both scorers.
3. **The generalist is not image-blind.** Its reports are study-specific on both scorers, and the image slightly improves which findings it names (CheXbert).
4. **Holistic agreement hides the generalist's weak image use.** On the judge, Lingshu's image-free text matches its real-image reports exactly (1.29 vs 1.29). Reference-based holistic evaluation can't tell what came from the image.
5. **The mechanism is consistent with population priors.** Image-free generalist text scores well against abnormal studies and poorly against normal ones. This is not yet tested across datasets.
6. **The headline between-model claim is only directional.** Source-following is larger for the specialist on both scorers, but not confirmed by the preregistered test.

---

## 5. Current standing

### Novelty

- **Prior work.** Language priors, medical VLM hallucination and the limits of reference metrics are established. Interventional audits (image swap, occlusion, text-only baselines) were published in 2026 for chest X-ray **question answering** (Lotfinia et al., arXiv 2606.17710; ModaLens, arXiv 2609.15635).
- **Our contribution.** Extending interventional auditing to **free-text report generation**, which the closest paper names as future work. Specifically:
  - two-reference scoring, against the source and the target,
  - a specialist positive control,
  - a test of how the conclusions depend on the scorer.
- **Not novel.** That VLMs rely on language priors, and image swapping in itself.

### Significance

- **Robust (both scorers, q \< 0.001):** the positive control, the specialist > generalist real-vs-blank gain, and the generalist's study specificity.
- **Scorer-dependent:** whether the generalist's image adds anything beyond its prior.
- **Not confirmed:** the preregistered primary outcome (judge p = 0.14; CheXbert p = 0.029, with the CI touching zero).

### Publishability

- **Now:** a realistic workshop or short paper at a medical imaging or ML-for-health venue. Estimated readiness about 6/10.
- **With MedGemma, a third scorer validated on ReXVal, and the tier analysis:** about 7/10.
- **With multiple datasets (the scale-up plan):** a stronger venue becomes plausible.
- **Timing risk:** the gap is publicly named, so posting a preprint early protects priority.

### Strengths

- Preregistered study, with an honestly reported primary null
- A positive control that works on both scorers
- Two independent scorers (holistic LLM and finding-level CheXbert)
- Deterministic, reproducible generation (23/23 exact reruns)
- Seeded, contrast-maximising pair selection; committed code and public study IDs
- Blinded judging; Benjamini–Hochberg correction for secondary outcomes
- A clear, timely gap acknowledged in the closest prior work

### Weaknesses

| Weakness | Status / plan |
| --- | --- |
| One dataset (MIMIC), 100 targets | Scale-up plan: IU X-ray and CheXpert Plus, about 300 targets each |
| Two models so far (third in progress) | Scale-up plan: 7–8 models, including the Qwen2.5-VL vs Lingshu ablation |
| No expert validation | Planned: validate scorers on ReXVal radiologist annotations; add RadGraph and GREEN |
| Preregistered primary null | Planned: a separately preregistered, properly powered Study 2 |
| Small LLM judge (8B) | Accepted limitation; no longer the only scorer |
| MedGemma in 8-bit | Accepted limitation; the precision applies equally to all conditions |
| MAIRA-2 needs an image (asymmetric no-input baseline) | Real vs blank used as the fair comparison between models |
| CheXbert flatters empty outputs on normal studies | Emphasise the abnormal stratum; RadGraph to add finer granularity |
| Possible training overlap with MIMIC | Test split used; contamination table planned |

### Open items

1. Complete MedGemma (generation, judge, CheXbert, analysis).
2. Add a deviation note to the Study 1 preregistration (no human verification).
3. RadGraph F1 scorer, then ReXVal validation of all scorers.
4. Finding-tier analysis from cached labels.
5. Begin the scale-up (see `project_a_scaleup_plan`).