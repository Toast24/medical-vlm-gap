"""Finding-level image dependence from stored CheXbert labels.
Per model and finding: image gain = recall(C1) - recall(no-info) among studies whose own reference has the finding;
source-following = P(finding in C2 output | only source has it) - P(finding in C2 output | only target has it).
Tiers by prevalence in the sample's target references: common >=25%, moderate 10-25%, rare <10%.
Usage: DS=scaled python evaluation/analyze_finding_tiers.py"""
import json, os
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; DS = os.environ.get("DS", "scaled")
MODELS = {"Lingshu": "lingshu", "MAIRA-2": "maira2", "MedGemma-27B": "medgemma", "Qwen2.5-VL-7B": "qwen", "CheXagent-2-3b": "chexagent"}
NOINFO = {"MAIRA-2": "C3_vs_own"}
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]

def load(tag):
    J = ROOT / f"results/project_a_chexbert_{DS}_{tag}"
    if not (J / "judge_shard0.jsonl").exists(): return None
    key = pd.read_csv(J / "key.csv", dtype=str).set_index("item_id")
    out = {}
    for r in map(json.loads, open(J / "judge_shard0.jsonl")):
        k = key.loc[r["item_id"]]; p = r["parsed"]
        out[(k.target_study_id, k.role)] = (set(p["output_findings"]), set(p["reference_findings"]))
    return out

rows, prev = [], None
for name, tag in MODELS.items():
    d = load(tag)
    if not d: continue
    studies = sorted({s for s, _ in d})
    if prev is None:
        own = [d[(s, "C1_vs_own")][1] for s in studies if (s, "C1_vs_own") in d]
        prev = {f: sum(f in r for r in own) / len(own) for f in FINDINGS}
    noinfo = NOINFO.get(name, "C5_vs_own")
    for f in FINDINGS:
        pos = [s for s in studies if (s, "C1_vs_own") in d and f in d[(s, "C1_vs_own")][1]]
        r1 = sum(f in d[(s, "C1_vs_own")][0] for s in pos) / len(pos) if pos else None
        r0 = sum(f in d[(s, noinfo)][0] for s in pos if (s, noinfo) in d) / len(pos) if pos else None
        neg = [s for s in studies if (s, "C1_vs_own") in d and f not in d[(s, "C1_vs_own")][1]]
        sp1 = sum(f not in d[(s, "C1_vs_own")][0] for s in neg) / len(neg) if neg else None
        sp0 = sum(f not in d[(s, noinfo)][0] for s in neg if (s, noinfo) in d) / len(neg) if neg else None
        ba1 = None if r1 is None or sp1 is None else (r1 + sp1) / 2
        ba0 = None if r0 is None or sp0 is None else (r0 + sp0) / 2
        src_only = [s for s in studies if (s, "C2_vs_source") in d and f in d[(s, "C2_vs_source")][1] and f not in d[(s, "C2_vs_target")][1]]
        tgt_only = [s for s in studies if (s, "C2_vs_target") in d and f in d[(s, "C2_vs_target")][1] and f not in d[(s, "C2_vs_source")][1]]
        ps = sum(f in d[(s, "C2_vs_source")][0] for s in src_only) / len(src_only) if src_only else None
        pt = sum(f in d[(s, "C2_vs_target")][0] for s in tgt_only) / len(tgt_only) if tgt_only else None
        rows.append(dict(model=name, finding=f, prevalence=prev[f], n_pos=len(pos), recall_img=r1, recall_noinfo=r0,
                         image_gain=None if r1 is None or r0 is None else r1 - r0, ba_img=ba1, ba_noinfo=ba0,
                         ba_gain=None if ba1 is None or ba0 is None else ba1 - ba0,
                         n_src_only=len(src_only), n_tgt_only=len(tgt_only),
                         source_following=None if ps is None or pt is None else ps - pt))
df = pd.DataFrame(rows)
df["tier"] = pd.cut(df.prevalence, [-0.01, 0.10, 0.25, 1.01], labels=["rare (<10%)", "moderate (10-25%)", "common (>=25%)"])
pd.set_option("display.width", 200)
print(f"dataset: {DS}\n\n== prevalence in target references ==")
print(df.drop_duplicates("finding").set_index("finding").prevalence.sort_values(ascending=False).round(2).to_string())
for col in ["ba_gain", "image_gain", "source_following"]:
    print(f"\n== {col} by finding (rows) and model (columns) ==")
    print(df.pivot(index="finding", columns="model", values=col).round(2).to_string())
    print(f"\n== {col}: mean by tier ==")
    print(df.groupby(["tier", "model"], observed=True)[col].mean().unstack().round(2).to_string())
out = ROOT / f"reports/finding_tiers_{DS}.csv"; df.to_csv(out, index=False); print(f"\nwrote {out.relative_to(ROOT) if hasattr(out, "relative_to") else out}")
summary = {"dataset": DS, "tier_rule": "prevalence in target references: rare <10%, moderate 10-25%, common >=25%",
           "metrics": {"ba_gain": "balanced accuracy (C1) minus balanced accuracy (no-info: C5, or C3 for MAIRA-2)",
                       "image_gain": "recall (C1) minus recall (no-info)",
                       "source_following": "P(finding in C2 output | only source has it) - P(... | only target has it)"},
           "prevalence": df.drop_duplicates("finding").set_index("finding").prevalence.round(3).to_dict(),
           "counts": df.drop_duplicates("finding").set_index("finding")[["n_pos", "n_src_only", "n_tgt_only"]].to_dict("index"),
           "by_finding": {c: {m: g.set_index("finding")[c].round(3).where(g.set_index("finding")[c].notna(), None).to_dict()
                              for m, g in df.groupby("model")} for c in ["ba_gain", "image_gain", "source_following"]},
           "by_tier": {c: {m: {str(t): (None if pd.isna(v) else round(float(v), 3)) for t, v in g.groupby("tier", observed=True)[c].mean().items()}
                           for m, g in df.groupby("model")} for c in ["ba_gain", "image_gain", "source_following"]}}
jout = ROOT / f"reports/finding_tiers_{DS}.json"; json.dump(summary, open(jout, "w"), indent=2); print(f"wrote {jout.relative_to(ROOT) if hasattr(jout, "relative_to") else jout}")
