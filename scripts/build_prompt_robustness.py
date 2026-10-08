"""Prompt-robustness subsets: 100 seeded IU contrast pairs, all 5 conditions, with alternative prompts.
Writes data/project_a_{iup1,iup2}_{pairs,conditions}.jsonl, results/project_a_{...}_source_references.jsonl,
and reports/prompt_robustness_subset.json (study IDs + prompts; public-safe)."""
import json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
PROMPTS = {"iup1": "Write the findings and impression sections of a radiology report for this chest X-ray.",
           "iup2": "What do you see in this chest radiograph?"}
pairs = [json.loads(l) for l in open(ROOT / "data/project_a_iu_pairs.jsonl")]
contrast = sorted([p for p in pairs if p["pair_type"] == "contrast"], key=lambda p: p["target_study_id"])
sub = random.Random(2026).sample(contrast, 100); ids = {p["target_study_id"] for p in sub}
cond = [json.loads(l) for l in open(ROOT / "data/project_a_iu_conditions.jsonl")]
srcs = {json.loads(l)["source_study_id"]: l for l in open(ROOT / "results/project_a_iu_source_references.jsonl")}
for ds, prompt in PROMPTS.items():
    with open(ROOT / f"data/project_a_{ds}_pairs.jsonl", "w") as f:
        for p in sub: f.write(json.dumps(dict(p, dataset=ds)) + "\n")
    rows = [dict(r, prompt=prompt, dataset=ds) for r in cond if r["target_study_id"] in ids]
    with open(ROOT / f"data/project_a_{ds}_conditions.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    with open(ROOT / f"results/project_a_{ds}_source_references.jsonl", "w") as f:
        for p in sub: f.write(srcs[p["source_study_id"]] if srcs[p["source_study_id"]].endswith("\n") else srcs[p["source_study_id"]] + "\n")
    print(f"{ds}: {len(rows)} condition rows, prompt: {prompt!r}")
json.dump({"seed": 2026, "n_pairs": 100, "source": "IU contrast pairs", "target_study_ids": sorted(ids),
           "prompts": {"original": cond[0]["prompt"], **PROMPTS}}, open(ROOT / "reports/prompt_robustness_subset.json", "w"), indent=2)
print("wrote reports/prompt_robustness_subset.json")
