import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[1]
import os
J = Path(os.environ.get("JUDGE_DIR", ROOT / "results/project_a_judge"))
norm = lambda x: str(x).strip().lstrip("sS").split(".")[0]
key = pd.read_csv(J / "key.csv", dtype=str)
items = {r["item_id"]: r for r in map(json.loads, open(J / "items.jsonl"))}

rows, fails = [], 0
for f in glob.glob(str(J / "judge_shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if not r.get("ok"): fails += 1; continue
        p = r["parsed"]; fs = p.get("output_findings", [])
        bad = [x for x in fs if x.get("status") == "contradicted" or (x.get("status") == "not_in_reference" and x.get("clinically_important"))]
        rows.append(dict(item_id=r["item_id"], specificity=p.get("specificity"), agreement=p.get("reference_agreement"),
                         n_findings=len(fs), n_supported=sum(x.get("status") == "supported" for x in fs),
                         n_contradicted=sum(x.get("status") == "contradicted" for x in fs),
                         major_unsupported=int(len(bad) > 0), n_missed=len(p.get("missed_reference_findings", [])),
                         rationale=p.get("rationale", "")))
j = pd.DataFrame(rows).drop_duplicates("item_id", keep="last") if rows else pd.DataFrame(columns=["item_id"])
df = key.merge(j, on="item_id", how="left")
num = ["specificity", "agreement", "major_unsupported", "n_findings", "n_supported", "n_contradicted", "n_missed"]
for c in num: df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
print(f"judged {df['agreement'].notna().sum()}/{len(df)} items ({fails} failed attempts logged)\n")
if df["agreement"].notna().sum() == 0: raise SystemExit("nothing judged yet")

by_role = df.groupby("role")[num].mean().round(2); by_role["n"] = df.groupby("role")["agreement"].count()
print("== means by role =="); print(by_role.to_string(), "\n")

def paired(a, b, col="agreement", seed=0):
    x = df[df.role == a].set_index("target_study_id")[col]; y = df[df.role == b].set_index("target_study_id")[col]
    d = (x - y).dropna()
    if len(d) < 3: return dict(comparison=f"{a} - {b}", metric=col, n=len(d))
    boot = np.random.default_rng(seed).choice(d.values, (10000, len(d))).mean(1)
    try: p = wilcoxon(d.values).pvalue
    except ValueError: p = float("nan")
    return dict(comparison=f"{a} - {b}", metric=col, n=len(d), mean_diff=round(d.mean(), 3),
                ci95=[round(np.percentile(boot, 2.5), 3), round(np.percentile(boot, 97.5), 3)],
                wilcoxon_p=round(p, 4), n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()), n_zero=int((d == 0).sum()))
tests = [paired("C2_vs_source", "C2_vs_target"), paired("C1_vs_own", "C1_vs_other"),
         paired("C1_vs_own", "C3_vs_own"), paired("C1_vs_own", "C4_vs_own"), paired("C1_vs_own", "C5_vs_own"),
         paired("C1_vs_own", "C3_vs_own", "specificity")]
print("== paired comparisons =="); [print(t) for t in tests]

core = df[df.role.isin(["C1_vs_own", "C2_vs_source"])]
rest = df[~df.role.isin(["C1_vs_own", "C2_vs_source"])]
rev = pd.concat([core, rest.sample(n=min(10, len(rest)), random_state=0)]).sample(frac=1, random_state=1)
rev = rev.assign(reference=rev.item_id.map(lambda i: items[i]["reference"]), output=rev.item_id.map(lambda i: items[i]["output"]))
rev = rev[["item_id", "reference", "output", "specificity", "agreement", "major_unsupported", "rationale"]]
for c in ["human_specificity", "human_agreement", "human_major_unsupported", "human_groundedness", "human_notes"]: rev[c] = ""
rev.to_csv(J / "human_review_blinded.csv", index=False)

tpl_p = ROOT / "results/project_a_manual_annotation.csv"
if tpl_p.exists() and "JUDGE_DIR" not in os.environ:
    tpl = pd.read_csv(tpl_p, dtype=str); tid = tpl.target_study_id.map(norm)
    g = lambda role, col: df[df.role == role].set_index("target_study_id")[col]
    for c, own in [("C1", "C1_vs_own"), ("C2", "C2_vs_target")]:
        tpl[f"{c}_image_specific"] = tid.map(g(own, "specificity"))
        tpl[f"{c}_reference_agreement"] = tid.map(g(own, "agreement"))
        tpl[f"{c}_major_unsupported"] = tid.map(g(own, "major_unsupported"))
    src, tgt = g("C2_vs_source", "agreement").align(g("C2_vs_target", "agreement"))
    tpl["C2_source_following"] = tid.map((src > tgt).astype(int) + ((src > tgt) & (src >= 2)).astype(int))
    tpl["notes"] = "judge-prefilled; groundedness/leakage for human review"
    tpl.to_csv(J / "manual_annotation_judge_prefilled.csv", index=False)

json.dump(dict(by_role=by_role.reset_index().to_dict("records"), tests=tests, failed_attempts=fails),
          open(J / "judge_summary.json", "w"), indent=2, default=str)
print(f"\nwrote to {J}: judge_summary.json, human_review_blinded.csv, manual_annotation_judge_prefilled.csv")
