# Preregistration: Project A scaled study

Committed before any model inference on the scaled sample. Any later deviation will be reported as such.

## Question
For free-text chest X-ray report generation, how much of a model's reference agreement depends on the supplied image?

## Sample
- 100 MIMIC-CXR test-split frontal target studies (70 abnormal, 30 normal), one per patient; pilot studies and patients excluded.
- Each target is paired with a source study from a different, non-target patient. Sources may be any study of such a patient, since the test split has only about 300 patients and allowing one study per patient for sources would leave too little choice for pairing.
- 80 maximum-contrast pairs (no shared findings; achieved mean Jaccard 0.00) and 20 same-label pairs (achieved mean Jaccard 0.87).
- Studies with any uncertain CheXpert label are excluded, which slightly favours unambiguous cases.
- Selection: `scripts/select_scaled_pairs.py`, seed 2026. Public study IDs: `data/project_a_scaled_public_manifest.jsonl`. Aggregate summary: `reports/project_a_scaled_selection_summary.json`.
- Size rationale: the pilot's between-model difference in source-following (+0.65, per-study SD about 1.7) needs about 60 pairs for 80% power. 100 gives margin.

## Models
Lingshu-7B (generalist), MAIRA-2 (chest X-ray specialist, positive control), and a third open medical VLM if available (MedGemma preferred). Decoding is greedy. Model-specific input adaptations are documented per model (e.g. a blank image substituted where an image is required).

## Conditions
C1 real image, C2 source image, C3 blank image, C4 metadata only, C5 no input. Prompt and decoding are fixed within each model across conditions.

## Outcomes
- **Primary:** source-following = S(C2, source reference) − S(C2, target reference), per study. Comparison: MAIRA-2 minus Lingshu, paired by study, on the 80 contrast pairs.
- **Secondary:**
  1. Within-model source-following (> 0?), per model and pair type.
  2. Own-versus-unrelated reference agreement for C1.
  3. Image-attributable agreement: C1 minus the no-information baseline (C5, or C3 for models that require an image), with real-vs-blank (C1 − C3) as the like-for-like comparison between models.
  4. All of the above stratified by normal/abnormal, view (PA/AP) and finding tier.
- **Scorers:** S = LLM-judge reference agreement (0–3; Llama-3.1-8B-Instruct, blinded, fixed prompt in `evaluation/run_judge.py`). The secondary scorer is CheXbert label agreement. Human verification covers a blinded subset (≥50 items), with agreement reported as Cohen's κ.

## Analysis
Paired, study-level. Mean differences with percentile-bootstrap 95% CIs (10,000 resamples, seed 0) and two-sided Wilcoxon signed-rank tests, α = 0.05 for the primary outcome. Secondary outcomes are reported with Benjamini–Hochberg correction. Items the judge fails to parse after retries are excluded and counted.

## Interpretation rules
- The positive control must work: MAIRA-2's within-model source-following CI on the contrast pairs must exclude 0. If it doesn't, the measurement is treated as insensitive and no claim is made about other models.
- A claim of weak image dependence for a model requires both near-zero source-following and C1 not clearly exceeding its no-information baseline, with agreement between the two scorers.
