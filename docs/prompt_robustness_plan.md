# Prompt-robustness check (exploratory; planned before any outputs)

**Question:** do the Study 2 findings (source-following gradient; prior share) depend on the prompt wording?
**Sample:** 100 IU X-ray contrast pairs (seed 2026; IDs in `reports/prompt_robustness_subset.json`), all 5 conditions.
**Models:** Lingshu-7B, Qwen2.5-VL-7B, MedGemma-27B (int8, 768-token limit), CheXagent-2-3b. MAIRA-2 excluded: it uses a fixed, model-defined prompt.
**Prompts:** original (Study 2); P1 report-style ("Write the findings and impression sections of a radiology report for this chest X-ray."); P2 minimal ("What do you see in this chest radiograph?"). Original-prompt outputs are reused (deterministic decoding).
**Outcomes:** per model and prompt, source-following on contrast pairs and prior share (C5/C1), scored with RadGraph (primary), judge and CheXbert.
**Robustness criterion:** the gradient is robust if, under each alternative prompt, MedGemma's source-following stays positive (RadGraph CI excluding 0) and exceeds Qwen's, and Lingshu's and Qwen's stay near zero.
