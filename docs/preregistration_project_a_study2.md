# Preregistration: Project A, Study 2 (multi-dataset, multi-model)

Committed before any Study 2 model output exists. Study 1 (MIMIC-CXR, n = 100) is reported separately and unchanged; it is not part of this confirmatory analysis. Runs of additional models on the Study 1 sample are exploratory.

## Question
In free-text chest X-ray report generation, do chest X-ray specialist models depend on the supplied image more than medical generalist models, and does generalists' image-free agreement track each population's abnormality base rate?

## Datasets and samples (selection: scripts/select_pairs_generic.py, seed 2026)
| Dataset | Pairs | Contrast / same-label | Normal targets | Notes |
|---|---|---|---|---|
| IU X-ray (OpenI) | 350 | 300 / 50 | 105 | 60% of pool normal; frontal view chosen by filename heuristic; no view metadata |
| CheXpert Plus, validation split | 100 | 85 / 15 | 16 | All 200 patients used; same-label pairs weakly matched (mean Jaccard 0.56); 83% AP |
Targets: one per patient. Sources: different, non-target patients. Labels for pairing: CheXbert on the reference reports (MIMIC used official CheXpert labels). Public IDs: `data/project_a_{iu,chexpert}_public_manifest.jsonl`; summaries: `reports/project_a_{iu,chexpert}_selection_summary.json`.

## Models
- Chest X-ray specialists: MAIRA-2, CheXagent-2-3b.
- Medical generalists: Lingshu-7B, MedGemma-27B (int8 language model, bf16 vision encoder).
- General-purpose base (ablation): Qwen2.5-VL-7B, run through the Lingshu runner with identical preprocessing and decoding.
Greedy decoding throughout. Model-specific adaptations (MAIRA-2 needs an image, so its C4/C5 use the blank image; CheXagent's default system prompt) are recorded per output row.

## Conditions
C1 real image, C2 source image, C3 512×512 black image, C4 metadata only, C5 no input. C4 text uses the same header as MIMIC, with each dataset's available fields (CheXpert: view position; IU: "not recorded", so IU's C4 is near-uninformative).

## Scorers
LLM judge (Llama-3.1-8B, blinded, prompt in evaluation/run_judge.py) and CheXbert label agreement; RadGraph F1 and GREEN if available before scoring. **The primary scorer is the one with the highest Kendall's τ against radiologists' clinically significant error counts on ReXVal, determined before any Study 2 output is scored.** The others are secondary.

## Outcomes
**Primary:** source-following, S(C2, source ref) − S(C2, target ref), on contrast pairs. Per study, the mean over specialists minus the mean over medical generalists, pooled across IU and CheXpert (n = 385 contrast pairs). Test: two-sided Wilcoxon signed-rank; percentile bootstrap 95% CI (10,000 resamples, stratified by dataset, seed 0); α = 0.05.

**Secondary** (Benjamini–Hochberg across all secondary tests):
1. The primary contrast within each dataset separately.
2. Within-model source-following, own-vs-unrelated, real-vs-blank (C1 − C3), and real-vs-no-information (C1 − C5, or C1 − C3 for MAIRA-2), per model and dataset.
3. **Base-rate test:** generalists' C5 agreement is higher on abnormal-majority datasets (CheXpert; MIMIC, from Study 1) than on IU; and, within each dataset, C5 agreement is higher on abnormal than on normal targets.
4. **Medical fine-tuning ablation:** Lingshu vs Qwen2.5-VL on all effects.
5. **Scorer dependence:** for each effect, whether primary and secondary scorers agree in sign and significance.
6. Strata: normal/abnormal, view (CheXpert only), pair type.

## Interpretation rules
- The positive control must pass per dataset: at least one specialist's within-model source-following CI on contrast pairs excludes 0. Otherwise that dataset's other results are treated as insensitive.
- Claims of weak image dependence for a model require near-zero source-following **and** C1 not exceeding its no-information baseline, on the primary scorer and at least one secondary scorer.
- Judge-parse failures are excluded and counted. Runs are resumed, not re-run, after interruptions.

## Known limitations (stated in advance)
No human expert validation (ReXVal used instead); 8B LLM judge; MedGemma in 8-bit; IU frontal-view heuristic; small, fully-used CheXpert pool; possible training-data overlap (contamination table to be reported).
