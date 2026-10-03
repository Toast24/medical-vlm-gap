"""Second, non-LLM scorer: CheXbert label agreement for every judge item.

Labels each generated output and reference with CheXbert ('rrg' mode: positive or uncertain -> 1) and
scores agreement as the Jaccard overlap of positive findings ('No Finding' excluded; both empty = 1.0).
Writes judge-format results to results/project_a_chexbert_scaled_<model>/ so that
evaluation/analyze_scaled.py runs with SCORER=chexbert. Labels are cached by text hash (git-ignored).
"""
import os, json, hashlib, shutil
from pathlib import Path
import torch
torch.set_num_threads(int(os.environ.get("CHEXBERT_THREADS", 4)))
from f1chexbert import F1CheXbert

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results/chexbert_labels_cache.json"
cache = json.load(open(CACHE)) if CACHE.exists() else {}
model = F1CheXbert(device="cpu")
NAMES = list(model.target_names)

def labels(text):
    k = hashlib.sha1(text.encode()).hexdigest()
    if k not in cache:
        try: v = model.get_label(text, mode="rrg")
        except TypeError: v = model.get_label(text)
        cache[k] = [int(x) for x in v]
    return cache[k]
pos = lambda v: {n for n, x in zip(NAMES, v) if x == 1 and n != "No Finding"}
jac = lambda a, b: 1.0 if not a and not b else len(a & b) / len(a | b)

for tag in ["lingshu", "maira2", "medgemma"]:
    J = ROOT / f"results/project_a_judge_scaled_{tag}"
    if not (J / "items.jsonl").exists(): print(f"{tag}: no items yet, skipped"); continue
    O = ROOT / f"results/project_a_chexbert_scaled_{tag}"; O.mkdir(parents=True, exist_ok=True)
    shutil.copy(J / "key.csv", O / "key.csv")
    items = [json.loads(l) for l in open(J / "items.jsonl")]
    with open(O / "judge_shard0.jsonl", "w") as f:
        for n, it in enumerate(items, 1):
            a, b = pos(labels(it["output"])), pos(labels(it["reference"]))
            f.write(json.dumps(dict(item_id=it["item_id"], ok=True, parsed=dict(
                reference_agreement=round(jac(a, b), 4), output_findings=sorted(a), reference_findings=sorted(b)))) + "\n")
            if n % 100 == 0:
                print(f"{tag}: {n}/{len(items)}", flush=True); json.dump(cache, open(CACHE, "w"))
    json.dump(cache, open(CACHE, "w"))
    print(f"{tag}: done, {len(items)} items -> {O}")
print(f"label cache: {len(cache)} unique texts")
