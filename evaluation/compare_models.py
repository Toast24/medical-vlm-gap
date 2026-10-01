import json, glob
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
MODELS = {"Lingshu": ROOT / "results/project_a_judge", "MAIRA-2": ROOT / "results/project_a_judge_maira2"}

def wide(J):
    key = pd.read_csv(J / "key.csv", dtype=str); rows = []
    for f in glob.glob(str(J / "judge_shard*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("ok"): rows.append(dict(item_id=r["item_id"], agreement=r["parsed"]["reference_agreement"]))
    df = key.merge(pd.DataFrame(rows).drop_duplicates("item_id"), on="item_id")
    return df.pivot_table(index="target_study_id", columns="role", values="agreement")

EFFECTS = {"source_following (C2 src - tgt)": ("C2_vs_source", "C2_vs_target"),
           "study_specificity (C1 own - other)": ("C1_vs_own", "C1_vs_other"),
           "image_vs_blank (C1 - C3)": ("C1_vs_own", "C3_vs_own"),
           "image_vs_no_info (C1 - C5)": ("C1_vs_own", "C5_vs_own")}
W = {m: wide(J) for m, J in MODELS.items()}
rng = np.random.default_rng(0)
print(f"{'effect':38s} {'Lingshu':>9s} {'MAIRA-2':>9s} {'diff (M-L)':>11s}  95% CI            n")
for name, (a, b) in EFFECTS.items():
    e = {m: (w[a] - w[b]) for m, w in W.items()}
    both = pd.concat(e, axis=1).dropna()
    d = both["MAIRA-2"] - both["Lingshu"]
    boot = rng.choice(d.values, (10000, len(d))).mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{name:38s} {both['Lingshu'].mean():+9.2f} {both['MAIRA-2'].mean():+9.2f} {d.mean():+11.2f}  [{lo:+.2f}, {hi:+.2f}]  {len(d)}")
print("\nraw C1 agreement with own reference:", {m: round(w['C1_vs_own'].mean(), 2) for m, w in W.items()})
print("C1 agreement with unrelated reference:", {m: round(w['C1_vs_other'].mean(), 2) for m, w in W.items()})
