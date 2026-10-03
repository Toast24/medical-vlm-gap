import json, random, hashlib, os
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
COND = Path(os.environ.get("JUDGE_CONDITIONS", ROOT / "data/project_a_pilot_conditions.jsonl"))
import os
RES = Path(os.environ.get("JUDGE_RESULTS", ROOT / "results/project_a_lingshu_pilot_clean.jsonl"))
SRC_REFS = Path(os.environ.get("JUDGE_SRC_REFS", ROOT / "results/project_a_source_references.jsonl"))
OUT = Path(os.environ.get("JUDGE_DIR", ROOT / "results/project_a_judge")); OUT.mkdir(parents=True, exist_ok=True)
load = lambda p: [json.loads(l) for l in open(p) if l.strip()]
norm = lambda x: str(x).strip().lstrip("sS").split(".")[0]
cond, res = load(COND), load(RES)

def pick(rows, cands):
    keys = set().union(*map(dict.keys, rows))
    return next(c for c in cands if c in keys)
kc_res = pick(res, ["condition", "condition_name", "condition_id"])
kt_res = pick(res, ["target_study_id", "study_id", "target_id"])
kout = pick(res, ["prediction", "output", "response", "generated_text", "generation", "text"])
print("results fields:", kc_res, kt_res, kout)

tref, src_of = {}, {}
for r in cond:
    t = norm(r["target_study_id"])
    if r.get("source_study_id"): src_of.setdefault(t, norm(r["source_study_id"]))
    if r["condition"][:2] == "C1" and r.get("reference_text"): tref[t] = r["reference_text"]
ext = {norm(r["source_study_id"]): r["reference_text"] for r in load(SRC_REFS)} if SRC_REFS.exists() else {}
sref = {t: ext[s] for t, s in src_of.items() if s in ext}
print(f"target refs: {len(tref)}/20, source refs: {len(sref)}/20")

outs = {(norm(r[kt_res]), str(r[kc_res])[:2]): r[kout] for r in res}
items, key = [], []
def add(t, c, text, ref, ref_study, role):
    if not text or not ref: print(f"[warn] missing for {role}"); return
    iid = "it" + hashlib.sha1(f"{t}|{role}".encode()).hexdigest()[:10]
    items.append(dict(item_id=iid, output=text, reference=ref))
    key.append(dict(item_id=iid, target_study_id=t, source_study_id=src_of.get(t), condition=c, ref_study=ref_study, role=role))
for t in sorted(tref):
    s = src_of.get(t)
    add(t, "C1", outs.get((t, "C1")), tref[t], t, "C1_vs_own")
    add(t, "C1", outs.get((t, "C1")), sref.get(t), s, "C1_vs_other")
    add(t, "C2", outs.get((t, "C2")), sref.get(t), s, "C2_vs_source")
    add(t, "C2", outs.get((t, "C2")), tref[t], t, "C2_vs_target")
    for c in ["C3", "C4", "C5"]: add(t, c, outs.get((t, c)), tref[t], t, f"{c}_vs_own")

order = list(range(len(items))); random.Random(1234).shuffle(order)
with open(OUT / "items.jsonl", "w") as f:
    for i in order: f.write(json.dumps(items[i]) + "\n")
pd.DataFrame([key[i] for i in order]).to_csv(OUT / "key.csv", index=False)
print(f"wrote {len(items)} items"); print(pd.DataFrame(key).role.value_counts().to_string())
