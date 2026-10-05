"""ReXVal: CheXbert label agreement (Jaccard of positive findings) for each candidate vs ground truth."""
import json, hashlib
from pathlib import Path
import pandas as pd, torch
torch.set_num_threads(4)
from f1chexbert import F1CheXbert
ROOT = Path(__file__).resolve().parents[1]; J = ROOT / "results/rexval_judge"
CACHE = ROOT / "results/chexbert_labels_cache.json"; cache = json.load(open(CACHE)) if CACHE.exists() else {}
m = F1CheXbert(device="cpu"); NAMES = list(m.target_names)
def lab(t):
    k = hashlib.sha1(t.encode()).hexdigest()
    if k not in cache:
        try: v = m.get_label(t, mode="rrg")
        except TypeError: v = m.get_label(t)
        cache[k] = [int(x) for x in v]
    return {n for n, x in zip(NAMES, cache[k]) if x == 1 and n != "No Finding"}
jac = lambda a, b: 1.0 if not a and not b else len(a & b) / len(a | b)
rows = [dict(item_id=it["item_id"], chexbert=jac(lab(it["output"]), lab(it["reference"]))) for it in map(json.loads, open(J / "items.jsonl"))]
json.dump(cache, open(CACHE, "w")); pd.DataFrame(rows).to_csv(ROOT / "results/rexval_chexbert.csv", index=False)
print(f"scored {len(rows)} candidates with CheXbert")
