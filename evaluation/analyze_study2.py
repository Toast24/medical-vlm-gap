"""Study 2 confirmatory analysis (docs/preregistration_project_a_study2.md, incl. Amendment 1 and primary-scorer decision).
Primary scorer: RadGraph. A hypothesis is SUPPORTED only if RadGraph and >=1 secondary (judge, CheXbert) are significant
in the predicted direction. Datasets: IU X-ray, CheXpert Plus (validation)."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, mannwhitneyu
ROOT = Path(__file__).resolve().parents[1]
DATASETS, SCORERS, PRIMARY = ["iu", "chexpert"], ["radgraph", "judge", "chexbert"], "radgraph"
MODELS = {"Lingshu": "lingshu", "MAIRA-2": "maira2", "MedGemma-27B": "medgemma", "Qwen2.5-VL-7B": "qwen", "CheXagent-2-3b": "chexagent"}
SPEC, GEN = ["MAIRA-2", "CheXagent-2-3b"], ["Lingshu", "MedGemma-27B"]
NOINFO = {"MAIRA-2": "C3_vs_own"}
NEED = {"C1_vs_own", "C1_vs_other", "C2_vs_source", "C2_vs_target", "C3_vs_own", "C4_vs_own", "C5_vs_own"}
norm = lambda x: str(x).strip().split(".")[0]
rng = np.random.default_rng(0)

def meta(ds):
    p = pd.DataFrame(map(json.loads, open(ROOT / f"data/project_a_{ds}_pairs.jsonl")))
    p["sid"] = p.target_study_id.map(norm)
    return p.set_index("sid")[["pair_type", "target_is_normal"]]

def wide(scorer, ds, tag):
    J = ROOT / f"results/project_a_{scorer}_{ds}_{tag}"
    if not (J / "key.csv").exists() or not glob.glob(str(J / "judge_shard*.jsonl")): return None
    key = pd.read_csv(J / "key.csv", dtype=str); rows = []
    for f in glob.glob(str(J / "judge_shard*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("ok"): rows.append(dict(item_id=r["item_id"], a=float(r["parsed"]["reference_agreement"])))
    if not rows: return None
    d = key.merge(pd.DataFrame(rows).drop_duplicates("item_id", keep="last"), on="item_id")
    w = d.pivot_table(index="target_study_id", columns="role", values="a"); w.index = w.index.map(norm)
    return w if NEED <= set(w.columns) and len(w) >= 50 else None

W, M = {}, {ds: meta(ds) for ds in DATASETS}
for sc in SCORERS:
    for ds in DATASETS:
        for m, tag in MODELS.items():
            w = wide(sc, ds, tag)
            if w is not None: W[(sc, ds, m)] = w.join(M[ds], how="left")
avail = sorted({m for (_, _, m) in W})
print("available (scorer, dataset) per model:", {m: sorted({(s, d) for (s, d, mm) in W if mm == m}) for m in avail})

def eff(w, m, name):
    if name == "sf": return w.C2_vs_source - w.C2_vs_target
    if name == "own_other": return w.C1_vs_own - w.C1_vs_other
    if name == "blank": return w.C1_vs_own - w.C3_vs_own
    if name == "noinfo": return w.C1_vs_own - w[NOINFO.get(m, "C5_vs_own")]
def pooled(sc, m, name, contrast=True):
    parts = []
    for ds in DATASETS:
        w = W.get((sc, ds, m))
        if w is None: continue
        s = eff(w, m, name); s = s[w.pair_type == "contrast"] if contrast else s
        parts.append(s.rename(lambda i: f"{ds}:{i}"))
    return pd.concat(parts).dropna() if parts else pd.Series(dtype=float)
def boot(d):
    ds = d.index.str.split(":").str[0]; groups = [d[ds == g].values for g in np.unique(ds)]
    b = [np.mean(np.concatenate([rng.choice(g, len(g)) for g in groups])) for _ in range(10000)]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]
def wtest(d, alt="two-sided"):
    try: return float(wilcoxon(d.values, alternative=alt).pvalue)
    except ValueError: return float("nan")
def holm(ps):
    ps = np.array(ps, float); o = np.argsort(ps); out = np.empty_like(ps); run = 0
    for k, i in enumerate(o): run = max(run, min(1, (len(ps) - k) * ps[i])); out[i] = run
    return out
def summ(d, alt="two-sided"):
    if len(d) < 5: return dict(n=len(d))
    return dict(n=len(d), mean=round(float(d.mean()), 4), ci95=boot(d), p=round(wtest(d, alt), 5))

res = {"primary_scorer": PRIMARY, "available_models": avail, "primary": {}, "hypotheses": {}, "secondary": [], "c5_by_dataset": {}}

# ---- Primary: specialists minus generalists, source-following, contrast pairs, pooled ----
for sc in SCORERS:
    sp = [m for m in SPEC if m in avail]; ge = [m for m in GEN if m in avail]
    if not sp or not ge: continue
    S = pd.concat({m: pooled(sc, m, "sf") for m in sp}, axis=1).mean(axis=1)
    G = pd.concat({m: pooled(sc, m, "sf") for m in ge}, axis=1).mean(axis=1)
    d = (S - G).dropna()
    res["primary"][sc] = dict(specialists=sp, generalists=ge, **summ(d))

# ---- Amendment hypotheses (each tested per scorer; Holm within hypothesis) ----
def h_a1(sc):
    ms = [m for m in ["MAIRA-2", "MedGemma-27B"] if m in avail]
    out = {m: summ(pooled(sc, m, "sf"), "greater") for m in ms}
    for m, q in zip(ms, holm([out[m].get("p", np.nan) for m in ms])): out[m]["p_holm"] = round(float(q), 5)
    return out
def h_a2(sc):
    if "Qwen2.5-VL-7B" not in avail: return {}
    ms = [m for m in ["MAIRA-2", "MedGemma-27B"] if m in avail]; q = pooled(sc, "Qwen2.5-VL-7B", "sf")
    out = {f"Qwen - {m}": summ((q - pooled(sc, m, "sf")).dropna(), "less") for m in ms}
    for k, p in zip(out, holm([v.get("p", np.nan) for v in out.values()])): out[k]["p_holm"] = round(float(p), 5)
    return out
def h_a3(sc):
    out = {}
    for m in [x for x in ["Lingshu", "MedGemma-27B"] if x in avail]:
        iu, cx = W.get((sc, "iu", m)), W.get((sc, "chexpert", m))
        tests = {}
        if iu is not None and cx is not None:
            tests["C5: IU < CheXpert"] = (iu.C5_vs_own.dropna(), cx.C5_vs_own.dropna())
        for ds, w in [("iu", iu), ("chexpert", cx)]:
            if w is not None:
                tests[f"C5 {ds}: normal < abnormal"] = (w.C5_vs_own[w.target_is_normal == True].dropna(), w.C5_vs_own[w.target_is_normal == False].dropna())
        r = {k: dict(n=[len(a), len(b)], mean=[round(float(a.mean()), 4), round(float(b.mean()), 4)],
                     p=round(float(mannwhitneyu(a, b, alternative="less").pvalue), 5)) for k, (a, b) in tests.items() if len(a) >= 5 and len(b) >= 5}
        for k, q in zip(r, holm([v["p"] for v in r.values()])): r[k]["p_holm"] = round(float(q), 5)
        out[m] = r
    return out
def h_a4(sc):
    if not {"Lingshu", "Qwen2.5-VL-7B"} <= set(avail): return {}
    d = (pooled(sc, "Lingshu", "blank", contrast=False) - pooled(sc, "Qwen2.5-VL-7B", "blank", contrast=False)).dropna()
    return {"Lingshu - Qwen (C1-C3)": summ(d, "greater")}
for name, fn in [("H-A1", h_a1), ("H-A2", h_a2), ("H-A3", h_a3), ("H-A4", h_a4)]:
    res["hypotheses"][name] = {sc: fn(sc) for sc in SCORERS}

def sig(x): return isinstance(x, dict) and x.get("p_holm", x.get("p", 1)) < 0.05
def verdict(hyp, path):
    get = lambda sc: (lambda d: [d := d.get(k, {}) if isinstance(d, dict) else {} for k in path][-1])(res["hypotheses"][hyp][sc])
    prim = sig(get(PRIMARY)); sec = any(sig(get(sc)) for sc in SCORERS if sc != PRIMARY)
    return "SUPPORTED" if prim and sec else "primary only" if prim else "secondary only" if sec else "not supported"

# ---- Secondary: per-model, per-dataset effects (two-sided), BH across all ----
for sc in SCORERS:
    for ds in DATASETS:
        for m in avail:
            w = W.get((sc, ds, m))
            if w is None: continue
            for e, contrast in [("sf", True), ("own_other", True), ("blank", False), ("noinfo", False)]:
                s = eff(w, m, e); s = (s[w.pair_type == "contrast"] if contrast else s).dropna()
                res["secondary"].append(dict(scorer=sc, dataset=ds, model=m, effect=e, **summ(s)))
ps = np.array([r.get("p", np.nan) for r in res["secondary"]], float); ok = ~np.isnan(ps)
if ok.sum():
    o = np.argsort(ps[ok]); r_ = ps[ok][o] * ok.sum() / (np.arange(ok.sum()) + 1); r_ = np.minimum.accumulate(r_[::-1])[::-1]
    q = np.empty_like(r_); q[o] = np.minimum(r_, 1); qs = np.full_like(ps, np.nan); qs[ok] = q
    for r, v in zip(res["secondary"], qs): r["q_bh"] = None if np.isnan(v) else round(float(v), 5)
for sc in SCORERS:
    res["c5_by_dataset"][sc] = {m: {ds: round(float(W[(sc, ds, m)].C5_vs_own.mean()), 4) for ds in DATASETS if (sc, ds, m) in W} for m in avail}

# ---- Report ----
print(f"\n== PRIMARY (specialists minus medical generalists; source-following; contrast pairs; IU+CheXpert) ==")
for sc, r in res["primary"].items(): print(f"  {sc:9s}{' (PRIMARY)' if sc == PRIMARY else '          '} {r}")
print("\n== AMENDMENT HYPOTHESES (verdict requires RadGraph + >=1 secondary) ==")
for m in res["hypotheses"]["H-A1"][PRIMARY]: print(f"  H-A1 {m}: {verdict('H-A1', [m])} | " + " | ".join(f"{sc} {res['hypotheses']['H-A1'][sc].get(m, {}).get('mean')} p_holm={res['hypotheses']['H-A1'][sc].get(m, {}).get('p_holm')}" for sc in SCORERS))
for k in res["hypotheses"]["H-A2"][PRIMARY]: print(f"  H-A2 {k}: {verdict('H-A2', [k])} | " + " | ".join(f"{sc} {res['hypotheses']['H-A2'][sc].get(k, {}).get('mean')} p_holm={res['hypotheses']['H-A2'][sc].get(k, {}).get('p_holm')}" for sc in SCORERS))
for m, tests in res["hypotheses"]["H-A3"][PRIMARY].items():
    for k in tests: print(f"  H-A3 {m} [{k}]: {verdict('H-A3', [m, k])} | " + " | ".join(f"{sc} means={res['hypotheses']['H-A3'][sc].get(m, {}).get(k, {}).get('mean')} p_holm={res['hypotheses']['H-A3'][sc].get(m, {}).get(k, {}).get('p_holm')}" for sc in SCORERS))
for k in res["hypotheses"]["H-A4"][PRIMARY]: print(f"  H-A4 {k}: {verdict('H-A4', [k])} | " + " | ".join(f"{sc} {res['hypotheses']['H-A4'][sc].get(k, {}).get('mean')} p={res['hypotheses']['H-A4'][sc].get(k, {}).get('p')}" for sc in SCORERS))
print("\n== C5 (no-input) agreement by dataset ==")
for sc in SCORERS: print(f"  {sc}: {res['c5_by_dataset'][sc]}")
sec = pd.DataFrame(res["secondary"])
if len(sec):
    sec["cell"] = sec.apply(lambda r: f"{r['mean']:+.3f} q={r['q_bh']}" if pd.notna(r.get("mean")) else f"n={r['n']}", axis=1)
    for sc in SCORERS:
        t = sec[(sec.scorer == sc) & (sec.effect == "sf")]
        if len(t): print(f"\n== source-following (contrast) by model x dataset [{sc}] =="); print(t.pivot(index="model", columns="dataset", values="cell").to_string())
json.dump(res, open(ROOT / "reports/study2_results.json", "w"), indent=2, default=str)
print("\nwrote reports/study2_results.json")
