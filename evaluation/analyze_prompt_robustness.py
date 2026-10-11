"""Prompt robustness (exploratory): source-following and prior share per model x prompt, on the same 100 IU contrast pairs.
Original-prompt results come from the main IU run, restricted to the subset. Includes a sensitivity check that
excludes studies whose C2 (swapped-image) report hit the token limit."""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[1]
SUB = set(str(i) for i in json.load(open(ROOT / "reports/prompt_robustness_subset.json"))["target_study_ids"])
PROMPTS = {"original": "iu", "P1 report-style": "iup1", "P2 minimal": "iup2"}
SCORERS = ["radgraph", "judge", "chexbert"]
MODELS = {"CheXagent-2-3b": "chexagent", "MedGemma-27B": "medgemma", "Lingshu": "lingshu", "Qwen2.5-VL-7B": "qwen"}
norm = lambda x: str(x).strip().lstrip("sS").split(".")[0]
rng = np.random.default_rng(0)

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
    w = w[w.index.isin(SUB)]
    return w if {"C1_vs_own", "C2_vs_source", "C2_vs_target", "C5_vs_own"} <= set(w.columns) and len(w) >= 50 else None

def truncated_c2(ds, tag):
    out, rf = set(), ROOT / f"results/project_a_{ds}_{tag}.jsonl"
    if rf.exists():
        for l in open(rf):
            r = json.loads(l)
            if r.get("hit_token_limit") and r["condition"].startswith("C2"): out.add(norm(r["target_study_id"]))
    return out

rows = []
for sc in SCORERS:
    for m, tag in MODELS.items():
        for pl, ds in PROMPTS.items():
            w = wide(sc, ds, tag)
            if w is None: continue
            sf = (w.C2_vs_source - w.C2_vs_target).dropna()
            b = rng.choice(sf.values, (5000, len(sf))).mean(1)
            try: p = float(wilcoxon(sf.values, alternative="greater").pvalue)
            except ValueError: p = float("nan")
            tr = truncated_c2(ds, tag) & set(sf.index); sf_nt = sf[~sf.index.isin(tr)]
            x = w[["C1_vs_own", "C5_vs_own"]].dropna()
            rows.append(dict(scorer=sc, model=m, prompt=pl, n=len(sf), sf=round(float(sf.mean()), 4),
                             sf_ci=[round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)], sf_p=round(p, 5),
                             n_c2_truncated=len(tr), sf_excl_truncated=round(float(sf_nt.mean()), 4) if len(sf_nt) else None,
                             prior_share=round(float(x.C5_vs_own.mean() / x.C1_vs_own.mean()), 3) if x.C1_vs_own.mean() > 0 else None))

df = pd.DataFrame(rows); pd.set_option("display.width", 220)
print("Prompt robustness (exploratory) - 100 IU contrast pairs\n")
for sc in SCORERS:
    t = df[df.scorer == sc]
    if not len(t): continue
    t = t.assign(cell=t.apply(lambda r: f"{r.sf:+.3f} [{r.sf_ci[0]:+.3f},{r.sf_ci[1]:+.3f}] p={r.sf_p}", axis=1),
                 nt=t.apply(lambda r: f"{r.sf_excl_truncated:+.3f} (excl. {r.n_c2_truncated})", axis=1))
    print(f"== source-following [{sc}] =="); print(t.pivot(index="model", columns="prompt", values="cell").reindex(columns=list(PROMPTS)).to_string())
    print(f"\n== source-following excluding truncated C2 reports [{sc}] =="); print(t.pivot(index="model", columns="prompt", values="nt").reindex(columns=list(PROMPTS)).to_string())
    print(f"\n== prior share C5/C1 [{sc}] =="); print(t.pivot(index="model", columns="prompt", values="prior_share").reindex(columns=list(PROMPTS)).to_string(), "\n")
json.dump(rows, open(ROOT / "reports/prompt_robustness_results.json", "w"), indent=2, default=str)
print("wrote reports/prompt_robustness_results.json")
