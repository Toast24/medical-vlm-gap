# Literature Review: Image Dependence, Language Priors, and Medical VLM Evaluation

## Project question

This project asks whether medical vision-language models actually use the image when generating a chest-X-ray report, or whether conventional reference-based evaluation can be satisfied substantially by learned language/population priors.

The controlled protocol keeps the target study, prompt, and decoding procedure fixed while changing the visual input:

- **C1 — real image:** correct image for the target study.
- **C2 — mismatched image:** another patient's image.
- **C3 — blank image:** visually uninformative input.
- **C4 — metadata only:** no usable image, but metadata/context.
- **C5 — no input:** no image.

The same outputs are evaluated against the correct reference and, where useful, deliberately wrong references.

The central distinction is:

> **Clinical correctness is not the same thing as image dependence, and reference agreement is not proof of visual grounding.**

---

## Executive conclusion

The literature strongly supports the motivation: language priors, hallucination, shortcut learning, modality attribution, and weaknesses of reference-based report metrics are all established concerns.

The more interesting potential novelty is narrower:

> **An intervention-based evaluation framework for free-text medical report generation that measures how much clinically meaningful report content survives removal or replacement of the image, while using wrong-reference controls to distinguish target-following from generic report similarity.**

The project should therefore **not** claim that medical VLMs using language priors is itself novel. The potential contribution is the combination of:

1. controlled image replacement,
2. image ablation,
3. source-vs-target reference scoring,
4. clinical finding-level evaluation,
5. information/specificity tiers,
6. and comparison between a generalist model and a specialist chest-X-ray model.

The closest literature must still be checked paper-by-paper for exact overlap before making a publication-level novelty claim.

---

# 1. Current experiment

Lingshu-7B has been run on 20 MIMIC-CXR test studies under five conditions:

| Condition | Input |
|---|---|
| C1 | Correct image |
| C2 | Mismatched image |
| C3 | Blank image |
| C4 | Metadata only |
| C5 | No input |

This produced **100 successful outputs**.

The current pilot uses effectively greedy decoding (`top_k=1`, repetition penalty 1.05). The seed field in the conditions file is not actually used. Target/source pairs were generated outside the repository and should currently be treated as effectively random.

### Current lexical result

- C1: 20 distinct outputs.
- C3: 19/20 duplicate outputs.
- C5: 19/20 duplicate outputs.
- C4: 15/20 duplicate outputs.

Thus the visual input clearly changes Lingshu's generated language, while image-free conditions collapse toward highly repetitive responses.

However, lexical variation does **not** establish clinical grounding.

### Current judge result

A text-only Llama-3.1-8B judge scored 139/140 output-reference pairs.

Approximate results:

- C1 vs target reference: **1.70**
- C5 vs target reference: **1.50**
- C1 − C5: **0.20**, p = 0.41
- C1 substantially exceeded C3/C4 in the current pilot.
- C2 showed little evidence of following the source study rather than the target study.

Interpretation:

> Lingshu changes its output in response to the visual input, but the current pilot does not establish that its clinically meaningful content is strongly image-dependent.

This is a low-powered result: n=20, coarse 0–3 judge scale, and a text-only LLM judge with limited resolution.

---

# 2. Foundational language-prior literature

## VQA v2 — Goyal et al.

**Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering**

https://arxiv.org/abs/1612.00837

Shows that standard VQA datasets contain strong language priors and introduced balanced data to reduce answer bias.

**Relevance:** foundational evidence that multimodal systems can answer correctly without relying strongly on visual evidence.

**Difference:** VQA rather than free-form medical report generation.

---

## VQA-CP — Agrawal et al.

**Don't Just Assume; Look and Answer: Overcoming Priors for Visual Question Answering**

https://arxiv.org/abs/1811.03347

Changes answer distributions between train and test to expose reliance on language priors.

**Relevance:** very close conceptually.

**Difference:** distribution shift rather than direct counterfactual replacement of the supplied image.

**Implication:** language-prior dependence is established; it should be motivation rather than the novelty claim.

---

# 3. Multimodal contribution / reliance

## Perceptual Score — Gat et al.

**Faithful Vision-Language Interpretation via Perceptual Score**

https://arxiv.org/abs/2109.07913

Introduces a framework for assessing whether multimodal outputs are actually affected by visual information.

**Relevance:** one of the closest methodological precedents.

**Difference:** not specifically designed for clinical report generation.

---

## MM-SHAP — Parcalabescu & Frank

**MM-SHAP: A Performance-agnostic Metric for Measuring Multimodal Contributions**

https://arxiv.org/abs/2303.13425

Uses SHAP-style attribution to quantify modality contributions.

**Relevance:** supports evaluating the contribution of each modality rather than assuming that supplying an image means it was used.

**Difference:** attribution-based rather than direct image intervention.

---

# 4. Medical VLM hallucination and trustworthiness

Recent medical VLM literature increasingly evaluates unsupported findings, hallucinations, robustness, and trustworthiness.

Useful search entry points:

- https://arxiv.org/search/?query=medical+vision+language+model+hallucination&searchtype=all
- https://scholar.google.com/scholar?q=medical+vision+language+model+hallucination+benchmark
- https://scholar.google.com/scholar?q=medical+vision+language+model+trustworthiness+benchmark

Relevant work includes ProbMed and CARES-style trustworthiness evaluations.

**Key distinction for this project:**

- Hallucination asks: *Is this statement supported?*
- Image dependence asks: *Would the model still produce this statement if the image were removed or replaced?*

The second question is the more distinctive target.

---

# 5. Radiology report-generation metrics

## RadGraph

**RadGraph: Extracting Clinical Entities and Relations from Radiology Reports**

https://arxiv.org/abs/2106.14463

Provides structured clinical entities and relations for comparing radiology reports.

**Relevance:** highly useful for our planned finding-level analysis.

**Limitation for our question:** it remains fundamentally reference-based.

---

## RadCliQ

**RadCliQ: A Generalized Metric for Evaluating Radiology Report Generation**

https://arxiv.org/abs/2203.16635

Moves report evaluation beyond simple lexical metrics.

**Relevance:** important current evaluation baseline.

**Limitation:** even a strong clinical reference metric does not establish that a generated finding came from the supplied image.

---

## GREEN

**GREEN: Generative Radiology Evaluation and Error Naming**

https://arxiv.org/abs/2405.03595

Uses a generative evaluator to identify and classify clinically meaningful report errors.

**Relevance:** useful for evaluating clinical correctness.

**Important distinction:** GREEN can assess whether an output is clinically wrong, but not necessarily whether that output would have been different without the image.

---

## ReXrank

Radiology report-generation benchmark/leaderboard:

https://github.com/rajpurkarlab/rexrank

Useful for understanding contemporary evaluation practice and baselines.

---

# 6. The two main models

## MAIRA-2

**MAIRA-2: Grounded Radiology Report Generation**

https://arxiv.org/abs/2406.04445

MAIRA-2 is a chest-X-ray specialist and is especially useful as a **positive control**.

The important comparison is not simply whether MAIRA-2 has a higher conventional score.

The useful question is whether it shows:

- strong source-following under C2,
- a large C1-C5 clinical-content gap,
- and stronger retention of high-specificity findings.

---

## Lingshu-7B

Lingshu-7B is the generalist medical VLM used in the current experiment.

Search/reference:

https://arxiv.org/search/?query=Lingshu-7B&searchtype=all

Its role is to test whether a capable generalist can generate clinically plausible reports whose apparent reference agreement exceeds measurable image dependence.

---

# 7. Shortcut learning in medical imaging

Chest-X-ray classification literature contains extensive evidence of non-pathological shortcuts:

- hospital/site effects,
- acquisition differences,
- portable/stationary imaging,
- devices,
- demographic correlations,
- and prevalence differences.

Useful search:

https://scholar.google.com/scholar?q=chest+x-ray+shortcut+learning+dataset+shift

This supports the motivation for C4 metadata-only testing.

However, most shortcut-learning work concerns classification rather than free-text report generation.

---

# 8. Missing-modality and modality-ablation literature

Useful search:

https://scholar.google.com/scholar?q=multimodal+medical+AI+modality+ablation+missing+modality

Common approaches include:

- modality dropout,
- modality masking,
- missing-modality robustness,
- modality permutation,
- modality attribution.

These support C3/C5 conceptually.

The project is different in emphasis:

> We are not merely asking whether the model remains functional without a modality. We want to quantify how much **clinically meaningful generated content** survives removal of the image.

---

# 9. Counterfactual / image-swap evaluation

Useful searches:

- https://scholar.google.com/scholar?q=counterfactual+evaluation+vision+language+models+image+swap
- https://scholar.google.com/scholar?q=mismatched+image+vision+language+model+medical
- https://scholar.google.com/scholar?q=shuffled+images+vision+language+medical

The broader VLM literature contains image permutation, negative-image, counterfactual image-text, and modality-replacement methods.

Therefore, **image swapping alone should not be claimed as novel**.

The stronger novelty question is:

> Has image swapping been systematically combined with image ablation, source-vs-target reference controls, and finding-level clinical analysis for free-text medical report generation?

That combination is the main literature gap to verify.

---

# 10. Retrieval and template baselines

MIMIC-CXR contains recurring clinical language and common findings. Therefore generic or retrieval-based reports can achieve nontrivial reference similarity.

Search:

- https://scholar.google.com/scholar?q=MIMIC-CXR+template+baseline+report+generation
- https://scholar.google.com/scholar?q=MIMIC-CXR+retrieval+baseline+report+generation

Relevant baselines include:

- fixed normal templates,
- frequent-finding templates,
- nearest-neighbour retrieval,
- report retrieval,
- language-only generation.

These are empirical estimates of the population prior.

C5 is especially interesting because it estimates this prior **inside the same trained multimodal model**.

---

# 11. Why conventional reference metrics are insufficient

| Metric | What it measures | What it does not prove |
|---|---|---|
| BLEU | n-gram overlap | image use |
| ROUGE | lexical overlap | image use |
| CIDEr | consensus similarity | image use |
| RadGraph F1 | clinical entities/relations | image provenance |
| RadCliQ | report quality | image provenance |
| GREEN | clinical errors | image provenance |
| LLM judge | semantic/clinical similarity | image provenance |
| CheXbert agreement | extracted findings | image provenance |

The proposed framework should therefore **not replace** these metrics.

Instead:

> Apply clinically meaningful metrics across controlled visual interventions.

---

# 12. The strongest potential novelty bubble

The most promising research direction is:

## Intervention-based clinical grounding evaluation for free-text medical VLM generation

The proposed framework:

1. keeps the target study and prompt fixed,
2. replaces the image,
3. removes the image,
4. scores the result against the target reference,
5. scores swapped-image outputs against both source and target references,
6. extracts clinical findings,
7. separates findings by specificity/information level,
8. compares a generalist model with a specialist positive control.

The important conceptual distinction is:

**Clinical correctness ≠ image dependence**

and:

**Reference agreement ≠ evidence that the image was used.**

---

# 13. Information tiers: a promising refinement

A binary "grounded / not grounded" label is probably too crude.

Instead classify generated clinical content into tiers.

### Tier 0 — Generic language

Examples:

- "The heart size is within normal limits."
- "No acute cardiopulmonary abnormality."
- generic report structure.

Likely to be strongly supported by language priors.

### Tier 1 — Common findings

Examples:

- cardiomegaly,
- mild bibasilar atelectasis,
- pleural effusion.

More informative but still common in MIMIC.

### Tier 2 — Specific findings

Examples:

- focal right lower-lobe opacity,
- specific pleural abnormality,
- specific device placement.

Expected to depend more strongly on the image.

### Tier 3 — Highly discriminative findings

Rare, anatomically specific, or study-specific findings.

These provide the strongest evidence of actual image use.

This makes the study more informative than a simple defect/no-defect evaluation.

---

# 14. Potential metric: Image-Conditional Clinical Information

A useful conceptual metric is:

\[
ICCI = S(C1,R_{target}) - S(C5,R_{target})
\]

where `S` is a clinically meaningful score.

A finding-level version could be:

\[
ICCI_k =
P(finding_k|C1)-P(finding_k|C5)
\]

This asks:

> Which types of clinical information disappear when the image disappears?

A source-following quantity for C2 could be:

\[
ImageFollow_k =
P(finding_k|C2)-P(finding_k|C5)
\]

with the C2 output additionally compared against the source and target studies.

These should initially be treated as proposed analysis quantities, not established metrics.

---

# 15. Wrong-reference control

For C2, calculate:

- output vs target reference,
- output vs source reference.

Then examine:

\[
S(C2,R_{source}) \quad vs \quad S(C2,R_{target})
\]

If source agreement is clearly larger, the model followed the supplied source image.

If source and target agreement are similar, the output may be dominated by generic language/population priors.

This is potentially one of the cleanest parts of the framework.

---

# 16. Experiments still needed

## A. Reproducibility

Repeat selected C1/C2 conditions.

Goal: quantify generation variance and determine whether observed effects are input-driven.

## B. Normal/abnormal stratification

Separate studies by clinical abnormality.

Goal: determine whether C5 scores well because common findings dominate the dataset.

## C. Human C2 source-following

Blind human comparison of:

- target reference,
- source reference,
- C2 output.

Goal: validate the LLM judge.

## D. MAIRA-2 positive control

Run the same 20 × 5 intervention matrix.

Goal: establish that the protocol can detect strong image dependence when it exists.

## E. CheXbert or equivalent structured metric

Compare extracted findings across C1–C5.

Goal: reduce dependence on the LLM judge.

## F. Information-tier analysis

Measure intervention effects separately for generic, common, specific, and highly discriminative findings.

## G. Scale-up

Only after the protocol demonstrates sensitivity.

Then expand to 100+ studies using a committed selection script and fixed seed.

---

# 17. Critical novelty questions

Before finalizing the paper, verify these individually:

1. Has any paper performed **image swapping** on medical report-generation VLMs?
2. Has any paper performed **blank/no-image ablation** on medical report-generation VLMs?
3. Has anyone combined swapping and ablation into one controlled matrix?
4. Has anyone scored swapped outputs against both **source and target references**?
5. Has anyone measured image dependence at the **clinical finding level**?
6. Has anyone separated generic/common findings from study-specific findings?
7. Has anyone compared a generalist medical VLM with a chest-X-ray specialist using the same intervention protocol?
8. Has anyone demonstrated high conventional report scores alongside low measured image dependence?

If questions 4–8 are largely unanswered, that is the most promising novelty area.

---

# 18. Recommended framing

Avoid:

> "Medical VLMs hallucinate because they don't look at images."

Too broad and unsupported by the current pilot.

Prefer:

> **"Reference agreement is not sufficient evidence of visual grounding in medical report generation."**

Then introduce controlled visual interventions as the measurement framework.

The central scientific question becomes:

> **When a medical VLM produces a clinically plausible report, how much of that report actually depends on the image it was given?**

---

# 19. Suggested paper structure

## Introduction

- Medical VLMs are increasingly evaluated using report-reference agreement.
- Reference agreement measures output similarity, not information provenance.
- Medical reports contain strong population and language priors.
- Introduce controlled visual interventions.
- Measure clinical correctness and image dependence separately.

## Related Work

- Language priors.
- Multimodal attribution.
- Medical VLM hallucination.
- Radiology report-generation metrics.
- Shortcut/counterfactual evaluation.

## Method

- Dataset.
- Models.
- Intervention matrix.
- Reference controls.
- Clinical finding extraction.
- Information tiers.
- Image-dependence metrics.

## Experiments

- Generalist model.
- Specialist positive control.
- Reproducibility.
- Normal/abnormal stratification.
- Human validation.

## Results

Primary result:

> Conventional reference agreement and measured image dependence can diverge.

Secondary result:

> The divergence can be localized by the specificity/information level of generated findings.

---

# 20. Final assessment

The established part of the literature is:

- models can exploit language priors,
- multimodal models can fail to use visual information,
- medical VLMs hallucinate,
- radiology report metrics have limitations,
- and modality attribution/ablation are established ideas.

The promising research gap is the **combination**:

> **image swap + image ablation + wrong-reference controls + clinical finding-level evaluation + information-tier analysis + specialist positive control**

for **free-text medical report generation**.

That is a sharper and more testable research question than simply asking whether a medical VLM uses language priors.

---

## Verification note

This document is a research map and proposal framing. Before using it as a final related-work section or making a publication-level novelty claim, each candidate paper should be individually checked against the exact protocol. Search-result pages are included where the canonical paper URL still needs confirmation; those should not be used as final bibliography entries.

The next literature-review pass should focus specifically on papers from **2022 onward** that combine:

- medical report generation,
- image swapping/permutation,
- missing/blank images,
- clinical finding extraction,
- and counterfactual evaluation.

That narrower search is the most likely route to establishing whether the proposed intervention matrix is genuinely underexplored.
