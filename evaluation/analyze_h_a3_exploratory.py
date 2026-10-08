"""EXPLORATORY (post hoc) reanalysis of H-A3, designed after the preregistered cross-dataset test was found to be
confounded by dataset reporting style. Two within-dataset analyses:
 A. Prior x population interaction: per study, (medical model C5) - (Qwen C5), both scored against the same reference.
    Prediction: positive on abnormal targets, negative on normal targets -> interaction (abnormal - normal) > 0.
 B. Prior share: mean no-info agreement / mean real-image agreement, per model and dataset (bootstrap CI).
Datasets: MIMIC (Study 1), IU X-ray, CheXpert. Scorers: RadGraph, judge, CheXbert."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"scaled": "MIMIC", "iu": "IU", "chexpert": "CheXpert"}
SCORERS = ["radgraph", "judge", "chexbert"]
MODELS = {"Lingshu": "lingshu", "MAIRA-2": "maira2", "MedGemma-27B": "medgemma", "Qwen2.5-VL-7B": "qwen", "CheXagent-2-3b": "chexagent"}
NOINFO = {"MAIRA-2": "C3_vs_own"}
norm = lambda x: str(x).strip().lstrip("sS").split(".")[0]
rng = np.random.default_rng(0)

def meta(ds):
    p = pd.DataFrame(map(json.loads, open(ROOT / f"data/project_a_{ds}_pairs.jsonl")))
    return p.assign(sid=p.target_study_id.map(norm)).set_index("sid")["target_is_normal"]
def wide(sc, ds, tag):
    J = ROOT / f"results/project_a_{sc}_{ds}_{tag}"
    if not (J / "key.csv").exists() or not glob.glob(str(J / "judge_shard*.jsonl")): return None
    key = pd.read_csv(J / "key.csv", dtype=str); rows = []
    for f in glob.glob(str(J / "judge_shard*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("ok"): rows.append(dict(item_id=r["item_id"], a=float(r["parsed"]["reference_agreement"])))
    if not rows: return None
    d = key.merge(pd.DataFrame(rows).drop_duplicates("item_id", keep="last"), on="item_id")
    w = d.pivot_table(index="target_study_id", columns="role", values="a"); w.index = w.index.map(norm)
    return w if {"C1_vs_own", "C3_vs_own", "C5_vs_own"} <= set(w.columns) and len(w) >= 50 else None

W, MET = {}, {ds: meta(ds) for ds in DATASETS}
for sc in SCORERS:
    for ds in DATASETS:
        for m, t in MODELS.items():
            w = wide(sc, ds, t)
            if w is not None: W[(sc, ds, m)] = w.join(MET[ds].rename("normal"), how="left")
res = {"note": "EXPLORATORY / post hoc; not part of the preregistered confirmatory tests", "A_interaction": [], "B_prior_share": []}

# ---- A: prior x population interaction ----
for sc in SCORERS:
    for ds, dname in DATASETS.items():
        q = W.get((sc, ds, "Qwen2.5-VL-7B"))
        if q is None: continue
        for m in ["Lingshu", "MedGemma-27B"]:
            w = W.get((sc, ds, m))
            if w is None: continue
            d = (w.C5_vs_own - q.C5_vs_own).dropna(); nrm = w.normal.reindex(d.index)
            ab, no = d[nrm == False], d[nrm == True]
            if len(ab) < 5 or len(no) < 5: continue
            boots = [rng.choice(ab.values, len(ab)).mean() - rng.choice(no.values, len(no)).mean() for _ in range(5000)]
            res["A_interaction"].append(dict(scorer=sc, dataset=dname, model=m, n_abnormal=len(ab), n_normal=len(no),
                mean_diff_abnormal=round(float(ab.mean()), 4), mean_diff_normal=round(float(no.mean()), 4),
                interaction=round(float(ab.mean() - no.mean()), 4),
                ci95=[round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
                p_one_sided=round(float(mannwhitneyu(ab, no, alternative="greater").pvalue), 5)))

# ---- B: prior share (no-info / real image) ----
for sc in SCORERS:
    for ds, dname in DATASETS.items():
        for m in MODELS:
            w = W.get((sc, ds, m))
            if w is None: continue
            x = w[["C1_vs_own", NOINFO.get(m, "C5_vs_own")]].dropna(); x.columns = ["c1", "c0"]
            if len(x) < 20 or x.c1.mean() <= 0: continue
            b = []
            for _ in range(3000):
                s = x.iloc[rng.integers(0, len(x), len(x))]
                if s.c1.mean() > 0: b.append(s.c0.mean() / s.c1.mean())
            res["B_prior_share"].append(dict(scorer=sc, dataset=dname, model=m, n=len(x),
                share=round(float(x.c0.mean() / x.c1.mean()), 3),
                ci95=[round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]))

A = pd.DataFrame(res["A_interaction"]); B = pd.DataFrame(res["B_prior_share"])
pd.set_option("display.width", 220)
print("EXPLORATORY (post hoc) H-A3 reanalysis\n")
print("== A. (medical model C5) - (Qwen C5): abnormal vs normal targets; interaction > 0 supports the population prior ==")
if len(A):
    A["cell"] = A.apply(lambda r: f"{r.interaction:+.3f} [{r.ci95[0]:+.3f},{r.ci95[1]:+.3f}] p={r.p_one_sided}", axis=1)
    for sc in SCORERS:
        t = A[A.scorer == sc]
        if len(t): print(f"\n[{sc}]"); print(t.pivot(index="model", columns="dataset", values="cell").to_string())
    print("\n(detail: mean difference on abnormal | normal targets)")
    print(A[["scorer", "dataset", "model", "n_abnormal", "n_normal", "mean_diff_abnormal", "mean_diff_normal"]].to_string(index=False))
print("\n== B. Prior share = no-info agreement / real-image agreement (higher = more of the score earned without the image) ==")
if len(B):
    B["cell"] = B.apply(lambda r: f"{r.share:.2f} [{r.ci95[0]:.2f},{r.ci95[1]:.2f}]", axis=1)
    for sc in SCORERS:
        t = B[B.scorer == sc]
        if len(t): print(f"\n[{sc}]"); print(t.pivot(index="model", columns="dataset", values="cell").reindex(columns=["MIMIC", "CheXpert", "IU"]).to_string())
json.dump(res, open(ROOT / "reports/study2_h_a3_exploratory.json", "w"), indent=2, default=str)
print("\nwrote reports/study2_h_a3_exploratory.json")
