"""ReXVal: Kendall's tau between each scorer's agreement and radiologists' mean error counts (200 candidates).
A good scorer has strongly NEGATIVE tau (higher agreement = fewer errors)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import kendalltau
ROOT = Path(__file__).resolve().parents[1]; J = ROOT / "results/rexval_judge"
key = pd.read_csv(J / "key.csv")
df = key.merge(pd.read_csv(ROOT / "results/rexval_chexbert.csv"), on="item_id", how="left")
jf = J / "judge_shard0.jsonl"
if jf.exists():
    j = {r["item_id"]: r["parsed"]["reference_agreement"] for r in map(json.loads, open(jf)) if r.get("ok")}
    df["judge"] = df.item_id.map(j)
rng = np.random.default_rng(0); res = {}
for s in [c for c in ["judge", "chexbert"] if c in df]:
    for target in ["sig_errors", "all_errors"]:
        d = df[[s, target]].dropna()
        tau = kendalltau(d[s], d[target]).statistic
        boots = [kendalltau(*d.iloc[rng.integers(0, len(d), len(d))].T.values).statistic for _ in range(2000)]
        res[f"{s} vs {target}"] = dict(n=len(d), tau=round(float(tau), 3), ci95=[round(float(np.nanpercentile(boots, 2.5)), 3), round(float(np.nanpercentile(boots, 97.5)), 3)])
for k, v in res.items(): print(f"{k:28s} tau={v['tau']:+.3f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  n={v['n']}")
json.dump(res, open(ROOT / "reports/rexval_validation.json", "w"), indent=2); print("wrote reports/rexval_validation.json")
