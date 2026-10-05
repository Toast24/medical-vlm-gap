"""ReXVal: build judge-format items (candidate vs ground truth) and per-candidate radiologist error counts."""
import json
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; RX = ROOT / "data/rexval"
c = pd.read_csv(RX / "50_samples_gt_and_candidates.csv")
r = pd.read_csv(RX / "6_valid_raters_per_rater_error_categories.csv")
cands = [x for x in ["radgraph", "bertscore", "s_emb", "bleu"] if x in c.columns]
print("candidate columns:", cands, "| rater candidate_type values:", sorted(r.candidate_type.unique()))
print("study_number range:", r.study_number.min(), "-", r.study_number.max(), "| clinically_significant values:", sorted(r.clinically_significant.unique()))
nums = sorted(r.study_number.unique()); ids = list(c.study_id)
idx = (lambda n: ids.index(n)) if set(nums) <= set(ids) else (lambda n: n - 1) if min(nums) == 1 else (lambda n: n)
cmap = {str(t): t for t in r.candidate_type.unique()}
missing = [k for k in cands if k not in cmap]
if missing: raise SystemExit(f"candidate names differ between files: {missing} vs {list(cmap)}")
sig = r[r.clinically_significant.astype(str).str.lower().isin(["true", "1", "yes"])]
per = lambda d: d.groupby(["study_number", "candidate_type", "rater_index"]).num_errors.sum().groupby(["study_number", "candidate_type"]).mean()
rad = pd.DataFrame({"sig_errors": per(sig), "all_errors": per(r)}).fillna(0).reset_index()
J = ROOT / "results/rexval_judge"; J.mkdir(parents=True, exist_ok=True)
items, key = [], []
for _, x in rad.iterrows():
    row = c.iloc[idx(x.study_number)]; iid = f"rx_{int(x.study_number)}_{x.candidate_type}"
    items.append(dict(item_id=iid, output=str(row[x.candidate_type]), reference=str(row.gt_report)))
    key.append(dict(item_id=iid, study_number=int(x.study_number), candidate_type=x.candidate_type,
                    sig_errors=x.sig_errors, all_errors=x.all_errors, role="rexval", target_study_id=str(int(x.study_number))))
with open(J / "items.jsonl", "w") as f:
    for it in items: f.write(json.dumps(it) + "\n")
pd.DataFrame(key).to_csv(J / "key.csv", index=False)
print(f"wrote {len(items)} items | mean clinically significant errors per candidate: {rad.sig_errors.mean():.2f} (range {rad.sig_errors.min():.1f}-{rad.sig_errors.max():.1f})")
