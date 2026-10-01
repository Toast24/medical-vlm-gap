# Literature Review: Image Dependence in Medical Report Generation

*Project A, medical-vlm-gap. Revised 1 October 2026. Every link in this document was checked on that date; see the verification table at the end.*

---

## 1. The question

When a medical vision-language model writes a chest X-ray report, how much of what it says depends on the image it was given?

Reference-based evaluation can't answer this. A report can agree with the radiologist's reference because the model read the image, or because it wrote a plausible report about a typical patient. On a hospital population like MIMIC-CXR, where most patients have common findings, those two routes can earn similar scores.

The project measures image dependence directly. The target study, prompt and decoding stay fixed while the visual input changes:

| Condition | Visual input | Purpose |
| --- | --- | --- |
| C1 Real | Correct image | Normal behaviour |
| C2 Mismatched | Another patient's image | Does the output follow the supplied image? |
| C3 Blank | Uniform black image | Behaviour with an uninformative image |
| C4 Metadata only | No image (or blank, if required), acquisition metadata as text | Influence of non-visual context |
| C5 No input | No image (or blank, if required) | The model's language-only prior |

Each output is scored against the correct reference and against references it should not match. C2 outputs are scored against both the source study's report and the target study's report. C1 outputs are also scored against an unrelated study's report, which gives a chance baseline.

The central distinction: **reference agreement is not evidence that the image was used.**

---

## 2. Summary of where the literature stands

The motivation is well established. Language priors in vision-language models, failures to use visual input, shortcut learning in chest X-ray models and the limits of reference-based report metrics are all documented.

Since June 2026, the core idea has also been published for **question answering**. Lotfinia et al. introduced a causal audit for chest X-ray question answering that swaps and occludes images and compares results with matched text-only baselines. They found that accuracy and image use come apart. Their paper explicitly names free-form report generation as the natural next step and leaves it untested.

The defensible contribution of this project is therefore narrower and clearer than originally framed:

> **Interventional auditing of free-text report generation.** This uses image swap and ablation, scores each output against both the correct reference and a wrong one, analyses findings by how guessable they are, and uses a chest X-ray specialist as a positive control.

The project should not claim that language-prior reliance in medical VLMs is new, nor that image swapping is new.

---

## 3. Current results (pilot, n = 20 MIMIC-CXR test studies)

**Setup.** There were 100 outputs per model, for Lingshu-7B (a generalist medical VLM) and MAIRA-2 (a chest X-ray specialist). Decoding was effectively greedy. Clinical content was scored by a blinded, text-only Llama-3.1-8B judge on a 0–3 agreement scale; 139 of 140 Lingshu items and most MAIRA-2 items were successfully scored.

**Reproducibility.** Rerunning 23 Lingshu cases reproduced every output exactly, so all differences between conditions are driven by the input.

**Lexical behaviour.** Lingshu's wording depends on the image: C1 gave 20 distinct outputs, while C3 and C5 collapsed to near-identical text. But its C1-vs-C2 token overlap (Jaccard 0.18) sits at its own cross-study floor (0.22), so lexical difference alone shows nothing about grounding.

**Clinical content, judged against references:**

| Measure | Lingshu-7B | MAIRA-2 |
| --- | --- | --- |
| C1 vs own reference | 1.70 | 1.47 |
| C1 vs unrelated reference | 1.26 | 0.63 |
| Source-following (C2: source minus target) | +0.05 (14/20 ties) | +0.70, 95% CI \[0.1, 1.3\], p = 0.052 |
| Own vs unrelated (C1) | +0.42, n.s. | +0.78, p = 0.035 |
| Real image vs blank (C1 − C3) | +0.90 | +0.89 |
| Real image vs no-information baseline (C1 − C5) | +0.20, p = 0.41 | +1.05, p = 0.002\* |

\*MAIRA-2 requires an image, so its C4 and C5 used the blank placeholder. Its C5 is therefore not a language-only baseline, and C1 − C5 is not a like-for-like comparison between the models.

**Stratification.** 19 of 20 target studies were abnormal. Lingshu's high C5 score therefore cannot come from matching normal studies. Its image-free output is a typical-patient report that earns partial credit across an abnormal population.

**Direct model comparison (paired by study, bootstrap):**

- Source-following: MAIRA-2 minus Lingshu = +0.65, CI \[−0.10, +1.45\]. Suggestive, not established.
- Real vs blank: −0.11. No difference.

**Interpretation.** Within Lingshu, the image changes the wording but adds little clinically measurable content beyond the language prior. Within MAIRA-2, the protocol detects clear image dependence, which validates the method. The difference between the models needs a larger sample: about 60 studies to detect the source-following difference with 80% power, so plan for about 100.

---

## 4. Closest prior work (2026)

### 4.1 Lotfinia et al. — the closest paper

**Vision-language models for chest radiography do not always need the image** (arXiv, June 2026) [https://arxiv.org/abs/2606.17710](https://arxiv.org/abs/2606.17710)

The study audits nine systems on 2,575 yes/no chest X-ray questions under four image conditions: original, swap to a different patient's same-label image, occlusion of the radiologist-marked region, and occlusion of an irrelevant region. It derives three behavioural metrics:

- **Causal grounding rate:** how often target occlusion flips a correct answer.
- **Unrelated-image answer rate:** how often a correct answer survives a swap.
- **Irrelevant-mask stability:** how often an answer survives an irrelevant occlusion.

Findings:

- A text-only model came within 5.7 accuracy points of the best multimodal system.
- Three systems ignored the image entirely.
- Image use was sparse and specific to particular findings.
- Grounding was weaker on AP (portable) studies.
- A text-only model could not be statistically distinguished from a reference radiologist on accuracy, yet never grounded an answer.

**Relevance:** this is the same conceptual move as Project A, applied to question answering.

**Difference:** it uses binary questions, not free-text reports. The authors state that grounding in open-ended generation may differ, and name extending the audit to generated reports as the natural next step.

**Ideas to adopt:**

- **Same-label vs opposite-label swaps.** They note opposite-label swaps as a complementary test they did not run. Project A's planned contrasting pairs are exactly this.
- **An irrelevant-change control** to estimate generic sensitivity.
- **Finding-level reporting.** In their data, cardiomegaly, consolidation, edema, effusion and pneumonia carried the grounding, while atelectasis and lung opacity were inert.
- **View-stratified analysis** (PA vs AP).
- **A radiologist reference read** on a subsample.

### 4.2 ModaLens

**ModaLens: Measuring Image Sensitivity in Report-Conditioned Medical VLMs** (arXiv, September 2026) [https://arxiv.org/abs/2609.15635](https://arxiv.org/abs/2609.15635) · paper page: [https://huggingface.co/papers/2609.15635](https://huggingface.co/papers/2609.15635)

The study runs a paired image-swap audit of MedGemma-27B on 3,199 MIMIC-CXR cases, replacing each image with one from another study (usually the same patient's) while keeping the question and report fixed. Providing the report sharply reduced sensitivity to the swap.

**Relevance:** it confirms that image-swap auditing on MIMIC-CXR is an active area.

**Difference:** it covers question answering with a report supplied as input, not report generation.

### 4.3 Context-conflict benchmarks

**MC-CXR: A Multi-Context Chest X-ray Benchmark for Context-Induced Disruption in Vision–Language Models** (arXiv, 2026) [https://arxiv.org/abs/2608.24118](https://arxiv.org/abs/2608.24118)

This benchmark tests whether a correct image-only decision survives conflicting text or a conflicting prior image. Misleading text pulled models far more often than misleading visual context. It is relevant to C4, and to any future condition that pairs a mismatched image with the target's metadata.

**Medical Context Distorts Decisions in Clinical Vision Language Models** (arXiv, 2026) [https://arxiv.org/abs/2605.17436](https://arxiv.org/abs/2605.17436)

This study evaluates general and medical VLMs on MIMIC-CXR tasks with added clinical context, and frames the results in terms of modality collapse toward language priors. It is relevant to C4.

### 4.4 Tangential 2026 work

**Auditing Medical Vision-Language Models on Chest Radiographs: Estimating Reference Agreement Across Institutions** (arXiv, 2026) [https://arxiv.org/abs/2608.07550](https://arxiv.org/abs/2608.07550)

This paper is about how well reference agreement transfers across institutions, not about image dependence. Cite it only if discussing the limits of reference agreement.

---

## 5. Foundational work on language priors

**Goyal et al., Making the V in VQA Matter** (CVPR 2017) [https://arxiv.org/abs/1612.00837](https://arxiv.org/abs/1612.00837)

The paper balanced VQA with complementary image pairs and showed that models exploit language priors. It is foundational motivation.

**Agrawal et al., Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering** (CVPR 2018) [https://arxiv.org/abs/1712.00377](https://arxiv.org/abs/1712.00377)

The paper introduced VQA-CP, in which answer priors differ between train and test, and showed that models degrade sharply under the shift. It probes priors through distribution shift, not through intervention on individual images.

**Tong et al., Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs** (CVPR 2024) *Cited from the reference list of Lotfinia et al.; no link verified here.*

---

## 6. Measuring modality contribution

**Gat, Schwartz & Schwing, Perceptual Score: What Data Modalities Does Your Model Perceive?** (NeurIPS 2021) [https://arxiv.org/abs/2110.14375](https://arxiv.org/abs/2110.14375)

The paper measures reliance on a modality by permuting that modality across test samples. It found that more accurate VQA models perceived the image less. This is a close methodological precedent: permuting images across samples is essentially C2 applied in aggregate.

**Parcalabescu & Frank, MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions in Vision and Language Models & Tasks** (ACL 2023) [https://arxiv.org/abs/2212.08158](https://arxiv.org/abs/2212.08158) · [https://aclanthology.org/2023.acl-long.223](https://aclanthology.org/2023.acl-long.223)

The paper introduces a Shapley-value score for modality contribution that does not depend on accuracy. It is attribution-based, whereas Project A is interventional, so cite it as a complementary approach.

---

## 7. Reliability probes for medical VLMs

**Yan et al., Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA** (Findings of ACL 2025; the ProbMed benchmark) [https://aclanthology.org/2025.findings-acl.981/](https://aclanthology.org/2025.findings-acl.981/)

The paper pairs questions with negated or hallucinated variants, and models fall below chance.

**Sepehri et al., MediConfusion: Can You Trust Your AI Radiologist?** (ICLR 2025) [https://openreview.net/forum?id=H9UnNgdq0g](https://openreview.net/forum?id=H9UnNgdq0g)

The benchmark uses image pairs that look different but that models confuse, constructed so that language priors alone cannot beat random. Even proprietary systems score below random.

*Both links are taken from the reference list of Lotfinia et al.*

**How this differs from Project A:** these benchmarks degrade performance through adversarial constructions. Project A asks whether ordinary, unmodified report content depends on the image.

---

## 8. Report-generation metrics

All of these are reference-based. They measure how well an output agrees with a reference, not where its content came from. The proposal is to apply them *across* interventions, not to replace them.

| Metric | Paper | Link |
| --- | --- | --- |
| CheXbert labels | Smit et al., EMNLP 2020 | [https://arxiv.org/abs/2004.09167](https://arxiv.org/abs/2004.09167) |
| RadGraph / RadGraph F1 | Jain et al., NeurIPS Datasets & Benchmarks 2021 | [https://arxiv.org/abs/2106.14463](https://arxiv.org/abs/2106.14463) |
| RadCliQ | Yu et al., *Evaluating progress in automatic chest X-ray radiology report generation*, Patterns 2023 | [https://doi.org/10.1101/2022.08.30.22279318](https://doi.org/10.1101/2022.08.30.22279318) · [https://pmc.ncbi.nlm.nih.gov/articles/PMC10499844](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499844) |
| GREEN | Ostmeier et al., *GREEN: Generative Radiology Report Evaluation and Error Notation*, Findings of EMNLP 2024 | [https://arxiv.org/abs/2405.03595](https://arxiv.org/abs/2405.03595) |
| ReXrank (leaderboard) | Zhang et al., 2024 | [https://arxiv.org/abs/2411.15122](https://arxiv.org/abs/2411.15122) · [https://rexrank.ai](https://rexrank.ai) |

For the scaled study, use CheXbert labels or RadGraph F1 as an objective second metric alongside the LLM judge. GREEN's error categories map naturally onto the "reference-unsupported finding" analysis.

---

## 9. Models

**Lingshu** (the generalist under test) LASA Team et al., *Lingshu: A Generalist Foundation Model for Unified Multimodal Medical Understanding and Reasoning*, 2025 [https://arxiv.org/abs/2506.07044](https://arxiv.org/abs/2506.07044) · [https://alibaba-damo-academy.github.io/lingshu/](https://alibaba-damo-academy.github.io/lingshu/)

**MAIRA-2** (the specialist, used as positive control) Bannur et al., *MAIRA-2: Grounded Radiology Report Generation*, 2024 [https://arxiv.org/abs/2406.04449](https://arxiv.org/abs/2406.04449)

MAIRA-2 takes structured inputs: frontal and lateral images, prior study and report, and the indication, technique and comparison sections. It cannot run without an image, which is why its C4 and C5 used the blank placeholder.

**Candidate third model: MedGemma** Sellergren et al., *MedGemma Technical Report* [https://arxiv.org/abs/2507.05201](https://arxiv.org/abs/2507.05201) *(taken from the reference list of Lotfinia et al.)*

Including MedGemma would make results directly comparable with both Lotfinia et al. and ModaLens.

---

## 10. Counterfactual training in report generation (not evaluation)

**Li et al., Contrastive Learning with Counterfactual Explanations for Radiology Report Generation (CoFE)** (ECCV 2024) [https://arxiv.org/abs/2407.14474](https://arxiv.org/abs/2407.14474)

CoFE builds counterfactual images by swapping patches between similar images with different diagnoses, and uses them to *train* report generators to avoid spurious features. It shares the counterfactual vocabulary but is a training method, not an audit, so cite it to head off reviewer confusion.

---

## 11. Shortcut learning in chest X-ray models

This motivates the metadata condition (C4) and view-stratified analysis. The following are cited from the reference list of Lotfinia et al.; their links have not been individually verified here.

- Geirhos et al., *Shortcut learning in deep neural networks*, Nature Machine Intelligence 2020.
- DeGrave, Janizek & Lee, *AI for radiographic COVID-19 detection selects shortcuts over signal*, Nature Machine Intelligence 2021.
- Zech et al., *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs*, PLoS Medicine 2018.

Most shortcut-learning work concerns classifiers, not free-text generation.

---

## 12. Gap analysis

| Question | Status in the literature (as of Oct 2026) |
| --- | --- |
| Do medical VLMs exploit language priors? | Established |
| Image swap to test image use in chest X-ray VLMs | Done for question answering (Lotfinia et al.; ModaLens) |
| Text-only baselines matched to multimodal models | Done for question answering (Lotfinia et al.) |
| Image swap and ablation applied to **free-text report generation** | **Not found**; named as future work by Lotfinia et al. |
| Swapped outputs scored against **both source and target references** | **Not found** |
| Image dependence measured **by finding tier** in generated reports | **Not found** for generation (finding-level results exist for question answering) |
| Generalist vs specialist on an identical intervention protocol for generation | **Not found** |
| High reference agreement shown alongside low image dependence in generation | **Not found**; shown for question answering accuracy |

This search was targeted, not exhaustive. Before submission, repeat it on arXiv, medRxiv and the MICCAI, MIDL and ML4H proceedings from 2025 onward. The space is moving fast, and a report-generation extension by another group is plausible.

---

## 13. Proposed contribution and quantities

**Two-reference scoring (the central method).** For C2, compute S(C2, R_source) − S(C2, R_target). Reading the image predicts a clearly positive value; reliance on the prior predicts about zero. This is what makes interventional auditing work for free text, where there is no single answer to flip.

**Agreement attributable to the image:**

ICCI = S(C1, R_target) − S(C_noinfo, R_target)

where C_noinfo is C5 for models that accept no image and C3 otherwise. This reports what the image contributes beyond the model's own prior. It should be presented as a proposed analysis quantity, not an established metric.

**Finding tiers.** Score findings separately by how guessable they are:

- **Tier 0:** generic language ("no acute cardiopulmonary process").
- **Tier 1:** common findings (cardiomegaly, effusion, atelectasis).
- **Tier 2:** specific findings (laterality, focal opacities, named devices and their position).
- **Tier 3:** rare or study-specific findings.

The prediction is that image dependence rises with tier, and that prior-driven models earn their agreement mostly in Tiers 0 and 1. This connects directly to the finding-level heterogeneity Lotfinia et al. report for question answering.

**Positive control.** A specialist model (MAIRA-2) must show the effects for a null result in another model to count as evidence.

---

## 14. Design for the scaled study

1. **Sample.** About 100 MIMIC-CXR test studies, chosen by a committed, seeded script with a mix of normal and abnormal cases.
2. **Pairs.** Deliberately contrasting pairs (opposite-label, differing devices and laterality), plus a smaller set of same-label pairs for comparability with Lotfinia et al.
3. **Models.** Lingshu-7B, MAIRA-2 and MedGemma (or another open generalist).
4. **Conditions.** C1–C5 as now. Add an irrelevant-change control and optionally gray and noise blanks, to test whether the black image is simply out of distribution.
5. **Metrics.** The LLM judge plus CheXbert labels or RadGraph F1. Report agreement between the two.
6. **Human validation.** A blinded subset of 50–100 items, with a second rater if possible, and κ reported.
7. **Preregistered primary outcome.** The difference in source-following between the models.
8. **Secondary analyses.** Results by finding tier, by view (PA vs AP), by normal vs abnormal, and the attributable-agreement quantity.

---

## 15. Recommended framing

Avoid:

> "Medical VLMs don't look at images."

This is too broad, already partly shown for question answering, and contradicted by MAIRA-2.

Prefer:

> **"Reference agreement is not evidence of visual grounding in medical report generation."**

Position the paper as **extending interventional auditing from question answering (Lotfinia et al., 2026) to free-text report generation**. The contribution is what generation requires that question answering does not: two-reference scoring, tiered findings, and the dissociation between agreement and image dependence.

**Suggested structure:**

- **Introduction:** reference agreement measures similarity, not where content came from; a summary of the question-answering audits; the gap in generation.
- **Related work:** language priors, modality contribution, medical VLM reliability probes, report metrics, 2026 interventional audits.
- **Method:** the intervention matrix, two-reference scoring, finding tiers, quantities, positive control.
- **Experiments:** three models, reproducibility, stratification, human validation.
- **Results:** agreement and image dependence diverge, and the divergence concentrates in guessable findings.

---

## 16. Link verification (1 October 2026)

| Item | Link | Status |
| --- | --- | --- |
| Lotfinia et al. 2026 | [https://arxiv.org/abs/2606.17710](https://arxiv.org/abs/2606.17710) | Verified (full text read) |
| ModaLens | [https://arxiv.org/abs/2609.15635](https://arxiv.org/abs/2609.15635) | Verified via paper page |
| MC-CXR | [https://arxiv.org/abs/2608.24118](https://arxiv.org/abs/2608.24118) | Verified |
| Medical Context Distorts Decisions | [https://arxiv.org/abs/2605.17436](https://arxiv.org/abs/2605.17436) | Verified |
| Auditing reference agreement across institutions | [https://arxiv.org/abs/2608.07550](https://arxiv.org/abs/2608.07550) | Verified |
| VQA v2 (Goyal) | [https://arxiv.org/abs/1612.00837](https://arxiv.org/abs/1612.00837) | Verified |
| VQA-CP (Agrawal) | [https://arxiv.org/abs/1712.00377](https://arxiv.org/abs/1712.00377) | Verified — **corrected** (was 1811.03347) |
| Perceptual Score (Gat) | [https://arxiv.org/abs/2110.14375](https://arxiv.org/abs/2110.14375) | Verified — **title and link corrected** |
| MM-SHAP | [https://arxiv.org/abs/2212.08158](https://arxiv.org/abs/2212.08158) | Verified — **corrected** (was 2303.13425) |
| CheXbert | [https://arxiv.org/abs/2004.09167](https://arxiv.org/abs/2004.09167) | Verified (newly added) |
| RadGraph | [https://arxiv.org/abs/2106.14463](https://arxiv.org/abs/2106.14463) | Verified |
| RadCliQ (Yu et al.) | [https://doi.org/10.1101/2022.08.30.22279318](https://doi.org/10.1101/2022.08.30.22279318) | Verified — **title and link corrected** (was 2203.16635) |
| GREEN | [https://arxiv.org/abs/2405.03595](https://arxiv.org/abs/2405.03595) | Verified — **title corrected** |
| ReXrank | [https://arxiv.org/abs/2411.15122](https://arxiv.org/abs/2411.15122) | Verified — **replaced** the unverified GitHub link |
| MAIRA-2 | [https://arxiv.org/abs/2406.04449](https://arxiv.org/abs/2406.04449) | Verified — **corrected** (was 2406.04445) |
| Lingshu | [https://arxiv.org/abs/2506.07044](https://arxiv.org/abs/2506.07044) | Verified (replaced search link) |
| CoFE | [https://arxiv.org/abs/2407.14474](https://arxiv.org/abs/2407.14474) | Verified (newly added) |
| ProbMed | [https://aclanthology.org/2025.findings-acl.981/](https://aclanthology.org/2025.findings-acl.981/) | From Lotfinia et al. reference list |
| MediConfusion | [https://openreview.net/forum?id=H9UnNgdq0g](https://openreview.net/forum?id=H9UnNgdq0g) | From Lotfinia et al. reference list |
| MedGemma report | [https://arxiv.org/abs/2507.05201](https://arxiv.org/abs/2507.05201) | From Lotfinia et al. reference list |
| Tong; Geirhos; DeGrave; Zech | — | Cited without links; verify before use |

The original version contained Google Scholar and arXiv search-result URLs in several places, and section 4 referred to "CARES-style" evaluations without a citation. Search URLs are not citable, so those entries were removed. Add CARES back with a verified link if needed.