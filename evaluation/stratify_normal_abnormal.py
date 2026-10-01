import json, re, glob
from pathlib import Path
import pandas as pd, numpy as np
from scipy.stats import wilcoxon
ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "results/project_a_judge"
STRATA = J / "study_strata.csv"
norm = lambda x: str(x).strip().lstrip("sS").split(".")[0]

NORMAL = r"no acute (cardiopulmonary|intrathoracic|process|abnormalit)|normal chest|no (evidence of )?acute disease|lungs are clear|no radiographic evidence of acute"
ABNORMAL = r"effusion|pneumothorax|consolidation|opacit|edema|cardiomegaly|enlarged|atelectasis|pneumonia|nodule|mass|fracture|tube|catheter|line |pacemaker|wire|device|congestion|infiltrat"
NEG = r"\b(no|without|resolved|negative for)\b[^.]{0,40}$"

def classify(text):
    t = text.lower()
    hits = []
    for m in re.finditer(ABNORMAL, t):
        before = t[max(0, m.start() - 60):m.start()]
        if not re.search(NEG, before): hits.append(m.group(0))
    return int(bool(hits)), ", ".join(sorted(set(hits)))[:60]

cond = [json.loads(l) for l in open(ROOT / "data/project_a_pilot_conditions.jsonl") if l.strip()]
ref = {norm(r["target_study_id"]): r["reference_text"] for r in cond if r["condition"].startswith("C1")}
if STRATA.exists():
    st = pd.read_csv(STRATA, dtype={"target_study_id": str}); print(f"using your edited {STRATA.name}")
else:
    st = pd.DataFrame([dict(target_study_id=t, abnormal=classify(x)[0], heuristic_hits=classify(x)[1]) for t, x in sorted(ref.items())])
    st.to_csv(STRATA, index=False); print(f"wrote heuristic labels to {STRATA} (edit the 'abnormal' column to correct)")

key = pd.read_csv(J / "key.csv", dtype=str)
rows = []
for f in glob.glob(str(J / "judge_shard*.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r.get("ok"): rows.append(dict(item_id=r["item_id"], agreement=r["parsed"]["reference_agreement"]))
df = key.merge(pd.DataFrame(rows), on="item_id")
wide = df.pivot_table(index="target_study_id", columns="role", values="agreement").reset_index()
wide = wide.merge(st[["target_study_id", "abnormal"]], on="target_study_id")
print(f"\nabnormal: {int(wide.abnormal.sum())}/{len(wide)} studies\n")

def cmp(g, a, b):
    d = (g[a] - g[b]).dropna()
    if len(d) < 3: return f"{a} - {b}: n={len(d)} (too few)"
    try: p = wilcoxon(d).pvalue
    except ValueError: p = float("nan")
    return f"{a} - {b}: n={len(d)}, mean {d.mean():+.2f}, +{(d>0).sum()}/-{(d<0).sum()}/={(d==0).sum()}, p={p:.3f}"
for lab, g in wide.groupby("abnormal"):
    print(f"== {'ABNORMAL' if lab else 'NORMAL'} studies (n={len(g)}) ==")
    print("  mean agreement:", {c: round(g[c].mean(), 2) for c in ["C1_vs_own", "C5_vs_own", "C3_vs_own", "C2_vs_source", "C2_vs_target"] if c in g})
    for a, b in [("C1_vs_own", "C5_vs_own"), ("C2_vs_source", "C2_vs_target"), ("C1_vs_own", "C1_vs_other")]:
        print("  " + cmp(g, a, b))
print("\nper-study labels:"); print(st.to_string(index=False))
