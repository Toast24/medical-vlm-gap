"""Label each manifest reference with CheXbert ('rrg' mode) -> labels CSV (study_id + 14 label columns, 1.0/0.0).
Usage: python label_manifest_chexbert.py <manifest.csv> <labels_out.csv>"""
import sys, json, hashlib
from pathlib import Path
import pandas as pd, torch
torch.set_num_threads(4)
from f1chexbert import F1CheXbert
ROOT = Path(__file__).resolve().parents[1]; CACHE = ROOT / "results/chexbert_labels_cache.json"
cache = json.load(open(CACHE)) if CACHE.exists() else {}
model = F1CheXbert(device="cpu"); NAMES = list(model.target_names)
man = pd.read_csv(sys.argv[1], dtype={"study_id": str})
out = []
for n, (sid, text) in enumerate(zip(man.study_id, man.reference_text), 1):
    k = hashlib.sha1(text.encode()).hexdigest()
    if k not in cache:
        try: v = model.get_label(text, mode="rrg")
        except TypeError: v = model.get_label(text)
        cache[k] = [int(x) for x in v]
    out.append({"study_id": sid, **{nm: float(x) for nm, x in zip(NAMES, cache[k])}})
    if n % 250 == 0: print(f"{n}/{len(man)}", flush=True); json.dump(cache, open(CACHE, "w"))
json.dump(cache, open(CACHE, "w"))
pd.DataFrame(out).to_csv(sys.argv[2], index=False)
lab = pd.DataFrame(out); pos = lab[[c for c in NAMES if c != "No Finding"]].sum(axis=1)
print(f"labelled {len(lab)} | no positive findings (normal): {(pos == 0).sum()} | abnormal: {(pos > 0).sum()}")
