"""Preregistered analysis for the Project A scaled study (docs/preregistration_project_a_scaled.md)."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[1]
MODELS = {"Lingshu": "lingshu", "MAIRA-2": "maira2", "MedGemma-27B": "medgemma"}
NOINFO = {"Lingshu": "C5_vs_own", "MAIRA-2": "C3_vs_own", "MedGemma-27B": "C5_vs_own"}  # MAIRA-2 needs an image
import os
SCORER = os.environ.get("SCORER", "judge")
print("scorer:", SCORER)
norm = lambda x: str(x).strip().lstrip("sS").split(".")[0]

pairs = pd.DataFrame(map(json.loads, open(ROOT / "data/project_a_scaled_pairs.jsonl")))
pairs["target_study_id"] = pairs.target_study_id.map(norm)
man = pd.read_csv(ROOT / "data/mimic-cxr-jpg/mimic_cxr_test_manifest.csv", dtype=str)
view = dict(zip(man.study_id.map(norm), man.primary_view.str.upper()))
meta = pairs.set_index("target_study_id")[["pair_type", "target_is_normal"]].assign(
    view=lambda d: d.index.map(view))

def wide(J):
    key = pd.read_csv(J / "key.csv", dtype=str); rows, fails = [], 0
    for f in glob.glob(str(J / "judge_shard*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("ok"): rows.append(dict(item_id=r["item_id"], a=r["parsed"]["reference_agreement"]))
            else: fails += 1
    df = key.merge(pd.DataFrame(rows).drop_duplicates("item_id", keep="last"), on="item_id")
    return df.pivot_table(index="target_study_id", columns="role", values="a"), fails, len(key)

W, info = {}, {}
for m, tag in MODELS.items():
    J = ROOT / f"results/project_a_{SCORER}_scaled_{tag}"
    if (J / "key.csv").exists() and glob.glob(str(J / "judge_shard*.jsonl")):
        w, fails, n = wide(J)
        need = {"C1_vs_own", "C1_vs_other", "C2_vs_source", "C2_vs_target", "C3_vs_own", "C4_vs_own", "C5_vs_own"}
        if not need <= set(w.columns) or len(w) < 50:
            print(f"{m}: incomplete ({len(w)} studies, roles {sorted(set(w.columns) & need)}), skipped"); continue
        W[m] = w.join(meta, how="left"); info[m] = dict(judged=int(w.notna().sum().sum()), items=n, failed=fails)
print("models:", info)

def effects(w, m):
    return {"source_following": w["C2_vs_source"] - w["C2_vs_target"],
            "own_vs_other": w["C1_vs_own"] - w["C1_vs_other"],
            "image_vs_blank": w["C1_vs_own"] - w["C3_vs_own"],
            "image_vs_noinfo": w["C1_vs_own"] - w[NOINFO[m]]}
def summ(d, seed=0):
    d = d.dropna()
    if len(d) < 3: return dict(n=len(d))
    boot = np.random.default_rng(seed).choice(d.values, (10000, len(d))).mean(1)
    try: p = float(wilcoxon(d.values).pvalue)
    except ValueError: p = float("nan")
    return dict(n=len(d), mean=round(d.mean(), 3), ci=[round(float(np.percentile(boot, 2.5)), 3), round(float(np.percentile(boot, 97.5)), 3)], p=p)
def bh(ps):
    ps = np.array(ps, float); ok = ~np.isnan(ps); q = np.full_like(ps, np.nan)
    if ok.sum():
        o = np.argsort(ps[ok]); r = ps[ok][o] * ok.sum() / (np.arange(ok.sum()) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]; tmp = np.empty_like(r); tmp[o] = np.minimum(r, 1); q[ok] = tmp
    return q

STRATA = {"all": lambda w: w.index == w.index, "contrast": lambda w: w.pair_type == "contrast",
          "same_label": lambda w: w.pair_type == "same_label", "normal": lambda w: w.target_is_normal == True,
          "abnormal": lambda w: w.target_is_normal == False, "PA": lambda w: w.view == "PA", "AP": lambda w: w.view == "AP"}

res = {"info": info, "primary": None, "within": [], "between": [], "means": {}}
if {"Lingshu", "MAIRA-2"} <= set(W):
    sf = {m: effects(W[m], m)["source_following"][W[m].pair_type == "contrast"] for m in ["Lingshu", "MAIRA-2"]}
    d = (sf["MAIRA-2"] - sf["Lingshu"]).dropna()
    res["primary"] = dict(outcome="source_following, MAIRA-2 minus Lingshu, contrast pairs", **summ(d))

for m, w in W.items():
    res["means"][m] = {c: round(float(w[c].mean()), 3) for c in w.columns if c.endswith(("_own", "_other", "_source", "_target"))}
    for eff, s in effects(w, m).items():
        for st, f in STRATA.items():
            res["within"].append(dict(model=m, effect=eff, stratum=st, **summ(s[f(w)])))
ms = list(W)
for i, a in enumerate(ms):
    for b in ms[i + 1:]:
        ea, eb = effects(W[a], a), effects(W[b], b)
        for eff in ea:
            for st in ["all", "contrast"]:
                d = (eb[eff][STRATA[st](W[b])] - ea[eff][STRATA[st](W[a])])
                res["between"].append(dict(comparison=f"{b} - {a}", effect=eff, stratum=st, **summ(d)))

sec = res["within"] + res["between"]
for r, q in zip(sec, bh([r.get("p", np.nan) for r in sec])): r["q_bh"] = None if np.isnan(q) else round(float(q), 4)
for r in sec:
    if isinstance(r.get("p"), float): r["p"] = round(r["p"], 4)
if res["primary"] and isinstance(res["primary"].get("p"), float): res["primary"]["p"] = round(res["primary"]["p"], 4)

print("\n== PRIMARY ==\n", res["primary"])
print("\n== mean agreement by role ==")
print(pd.DataFrame(res["means"]).round(2).to_string())
wt = pd.DataFrame(res["within"])
print("\n== within-model effects (mean [CI], BH q) ==")
wt["cell"] = wt.apply(lambda r: f"{r['mean']:+.2f} [{r['ci'][0]:+.2f},{r['ci'][1]:+.2f}] q={r['q_bh']}" if "mean" in r and isinstance(r["ci"], list) else f"n={r['n']}", axis=1)
for eff in ["source_following", "own_vs_other", "image_vs_blank", "image_vs_noinfo"]:
    print(f"\n-- {eff} --"); print(wt[wt.effect == eff].pivot(index="stratum", columns="model", values="cell").reindex(list(STRATA)).to_string())
print("\n== between-model ==")
for r in res["between"]: print(r)
out = ROOT / ("reports/project_a_scaled_results.json" if SCORER == "judge" else f"reports/project_a_scaled_results_{SCORER}.json")
json.dump(res, open(out, "w"), indent=2, default=str); print(f"\nwrote {out}")
