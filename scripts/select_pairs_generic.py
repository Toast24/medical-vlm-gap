"""Seeded contrast-pair selection for any dataset with a manifest + labels CSV (same rules as select_scaled_pairs.py).
Usage: python select_pairs_generic.py --dataset iu --manifest ... --labels ... [--n-targets 350 --n-same 50]"""
import argparse, json, random
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--labels", required=True)
ap.add_argument("--n-targets", type=int, default=350); ap.add_argument("--n-normal", type=int, default=-1)
ap.add_argument("--n-same", type=int, default=50); ap.add_argument("--top-k", type=int, default=10); ap.add_argument("--seed", type=int, default=2026)
a = ap.parse_args(); rng = random.Random(a.seed)
FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
            "Lung Opacity", "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices"]
man = pd.read_csv(a.manifest, dtype={"study_id": str, "subject_id": str})
lab = pd.read_csv(a.labels, dtype={"study_id": str}).drop_duplicates("study_id").set_index("study_id")
pos = {s: frozenset(f for f in FINDINGS if row[f] == 1.0) for s, row in lab[FINDINGS].iterrows()}
man = man[man.study_id.isin(pos)].sort_values("study_id")
rows = man.to_dict("records"); rng.shuffle(rows)
seen, pool = set(), []
for r in rows:
    if r["subject_id"] not in seen: seen.add(r["subject_id"]); pool.append(r)
by_sid = {r["study_id"]: r for r in rows}
norm_pool = sorted(r["study_id"] for r in pool if not pos[r["study_id"]])
abn_pool = sorted(r["study_id"] for r in pool if pos[r["study_id"]])
n_t = min(a.n_targets, len(pool) // 2)
n_norm = min(a.n_normal if a.n_normal >= 0 else round(0.3 * n_t), len(norm_pool))
n_abn = min(n_t - n_norm, len(abn_pool)); n_t = n_norm + n_abn
targets = rng.sample(norm_pool, n_norm) + rng.sample(abn_pool, n_abn)
same_set = set(rng.sample([t for t in targets if pos[t]], min(a.n_same, n_abn)))
tpids = {by_sid[t]["subject_id"] for t in targets}
src_pool = [s for s in by_sid if s not in set(targets) and by_sid[s]["subject_id"] not in tpids]
if len(src_pool) < n_t: raise SystemExit(f"source pool {len(src_pool)} < targets {n_t}")
jac = lambda x, y: 1.0 if not x and not y else len(x & y) / len(x | y)
con = lambda x, y: (1 - jac(x, y)) + 0.5 * (("Support Devices" in x) != ("Support Devices" in y))
used, pairs = set(), []
order = targets[:]; rng.shuffle(order); order = [t for t in order if t in same_set] + [t for t in order if t not in same_set]
for t in order:
    c = [s for s in src_pool if s not in used]; rng.shuffle(c)
    if t in same_set: ptype = "same_label"; c.sort(key=lambda s: (-jac(pos[t], pos[s]), con(pos[t], pos[s])))
    else: ptype = "contrast"; c.sort(key=lambda s: -con(pos[t], pos[s]))
    s = rng.choice(c[:a.top_k]); used.add(s); T, S = by_sid[t], by_sid[s]
    pairs.append(dict(dataset=a.dataset, target_study_id=int(t), target_subject_id=int(T["subject_id"]), target_image_path=T["image_path"],
                      target_view=T["view"], source_study_id=int(s), source_subject_id=int(S["subject_id"]), source_image_path=S["image_path"],
                      reference_text=T["reference_text"], reference_method=T["reference_method"],
                      source_reference_text=S["reference_text"], source_reference_method=S["reference_method"], seed=a.seed,
                      pair_type=ptype, target_is_normal=not pos[t], target_labels=sorted(pos[t]), source_labels=sorted(pos[s]),
                      jaccard=round(jac(pos[t], pos[s]), 3), contrast=round(con(pos[t], pos[s]), 3)))
pairs.sort(key=lambda p: p["target_study_id"]); ds = a.dataset
with open(ROOT / f"data/project_a_{ds}_pairs.jsonl", "w") as f:
    for p in pairs: f.write(json.dumps(p) + "\n")
with open(ROOT / f"data/project_a_{ds}_public_manifest.jsonl", "w") as f:
    for p in pairs: f.write(json.dumps({k: p[k] for k in ["dataset", "target_study_id", "source_study_id", "pair_type", "seed"]}) + "\n")
with open(ROOT / f"results/project_a_{ds}_source_references.jsonl", "w") as f:
    for p in pairs: f.write(json.dumps(dict(source_study_id=p["source_study_id"], reference_text=p["source_reference_text"], reference_method=p["source_reference_method"])) + "\n")
df = pd.DataFrame(pairs)
summary = dict(dataset=ds, seed=a.seed, requested_targets=a.n_targets, n_pairs=len(pairs), pool=len(pool), pool_normal=len(norm_pool),
               pool_abnormal=len(abn_pool), source_pool=len(src_pool), normal_targets=int(df.target_is_normal.sum()),
               pair_types=df.pair_type.value_counts().to_dict(),
               mean_jaccard={k: round(v, 3) for k, v in df.groupby("pair_type").jaccard.mean().items()},
               views=df.target_view.value_counts().to_dict())
json.dump(summary, open(ROOT / f"reports/project_a_{ds}_selection_summary.json", "w"), indent=2); print(json.dumps(summary, indent=2))
