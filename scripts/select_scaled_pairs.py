#!/usr/bin/env python3
"""Seeded, contrast-maximising target/source pair selection (Project A, scaled).

Dataset-agnostic: needs a manifest CSV (one frontal image + report path per study),
a study-level labels CSV, and a reports archive. Defaults point at MIMIC-CXR-JPG.
Private outputs (MIMIC-derived) go to data/ (git-ignored); aggregate outputs to reports/.
"""
import argparse, json, random, re, zipfile
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default=str(ROOT / "data/mimic-cxr-jpg/mimic_cxr_test_manifest.csv"))
ap.add_argument("--labels", default=str(ROOT / "data/mimic-cxr-jpg/mimic-cxr-2.0.0-chexpert.csv.gz"))
ap.add_argument("--reports-zip", default=str(ROOT / "data/mimic-cxr-jpg/mimic-cxr-reports.zip"))
ap.add_argument("--image-root", default=str(ROOT / "data/mimic-cxr-jpg"))
ap.add_argument("--url-list", default=str(ROOT / "data/mimic-cxr-jpg/selected_image_urls.txt"),
                help="previous download list, used only to infer the URL base")
ap.add_argument("--url-base", default=None)
ap.add_argument("--exclude", nargs="*", default=[str(ROOT / "data/project_a_pilot_20.jsonl")])
ap.add_argument("--n-targets", type=int, default=100)
ap.add_argument("--n-normal", type=int, default=30)
ap.add_argument("--n-same", type=int, default=20)
ap.add_argument("--top-k", type=int, default=10)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--name", default="project_a_scaled")
a = ap.parse_args()
rng = random.Random(a.seed)

FINDINGS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum",
            "Fracture", "Lung Lesion", "Lung Opacity", "Pleural Effusion", "Pleural Other",
            "Pneumonia", "Pneumothorax", "Support Devices"]
sid = lambda x: str(x).strip().lstrip("sS").split(".")[0]
pid = lambda x: str(x).strip().lstrip("pP").split(".")[0]

# ---- candidate pool: frontal studies with clean labels ----
man = pd.read_csv(a.manifest, dtype=str)
man = man[man.primary_view.str.upper().isin(["PA", "AP"])].copy()
man["sid"], man["pid"] = man.study_id.map(sid), man.subject_id.map(pid)
lab = pd.read_csv(a.labels)
lab["sid"] = lab.study_id.map(sid)
lab = lab[lab.sid.isin(set(man.sid))].drop_duplicates("sid").set_index("sid")
uncertain = (lab[FINDINGS] == -1.0).any(axis=1)
lab = lab[~uncertain]
pos = {s: frozenset(f for f in FINDINGS if row[f] == 1.0) for s, row in lab[FINDINGS].iterrows()}
normal = {s for s in lab.index if lab.at[s, "No Finding"] == 1.0 and not pos[s]}
man = man[man.sid.isin(pos)]

excl_s, excl_p = set(), set()
for f in a.exclude:
    if Path(f).exists():
        for r in map(json.loads, open(f)):
            excl_s |= {sid(r.get("target_study_id")), sid(r.get("source_study_id"))}
            excl_p |= {pid(r.get("target_subject_id")), pid(r.get("source_subject_id"))}
man = man[~man.sid.isin(excl_s) & ~man.pid.isin(excl_p)]

# ---- reports: require a usable findings/impression section ----
zf = zipfile.ZipFile(a.reports_zip)
zindex = {m.group(1): n for n in zf.namelist() if (m := re.search(r"/s(\d+)\.txt$", n))}
def extract(txt):
    sec = {}
    for name in ["FINDINGS", "IMPRESSION"]:
        m = re.search(rf"{name}:\s*(.*?)(?=\n\s*[A-Z][A-Z /()]+:|\Z)", txt, re.S)
        if m and m.group(1).strip(): sec[name] = " ".join(m.group(1).split())
    if "FINDINGS" in sec and "IMPRESSION" in sec:
        return f"FINDINGS: {sec['FINDINGS']} IMPRESSION: {sec['IMPRESSION']}", "findings+impression"
    if "IMPRESSION" in sec: return sec["IMPRESSION"], "impression"
    if "FINDINGS" in sec: return sec["FINDINGS"], "findings"
    return None, "narrative_fallback"
refs = {}
for s in man.sid:
    if s in zindex:
        text, method = extract(zf.read(zindex[s]).decode(errors="ignore"))
        if text and len(text) >= 20: refs[s] = (text, method)
man = man[man.sid.isin(refs)]

# one study per patient, deterministic order before seeded shuffle
man = man.sort_values("sid")
rows = man.to_dict("records"); rng.shuffle(rows)
seen, pool = set(), []
for r in rows:
    if r["pid"] not in seen: seen.add(r["pid"]); pool.append(r)
by_sid = {r["sid"]: r for r in rows}
pool_sids = [r["sid"] for r in pool]
norm_pool = sorted(s for s in pool_sids if s in normal)
abn_pool = sorted(s for s in pool_sids if pos[s])
print(f"pool after filters: {len(pool)} studies ({len(norm_pool)} normal, {len(abn_pool)} abnormal)")

n_abn = a.n_targets - a.n_normal
if len(norm_pool) < a.n_normal or len(abn_pool) < n_abn + a.n_targets:
    raise SystemExit("pool too small for requested sample; lower --n-targets or relax filters")
targets = rng.sample(norm_pool, a.n_normal) + rng.sample(abn_pool, n_abn)
same_set = set(rng.sample([t for t in targets if pos[t]], a.n_same))
target_pids = {by_sid[t]["pid"] for t in targets}
src_pool = [s for s in by_sid if s not in set(targets) and by_sid[s]["pid"] not in target_pids]

def jac(x, y): return 1.0 if not x and not y else len(x & y) / len(x | y)
def contrast(x, y): return (1 - jac(x, y)) + 0.5 * (("Support Devices" in x) != ("Support Devices" in y))

used, pairs = set(), []
order = targets[:]; rng.shuffle(order)
order = [t for t in order if t in same_set] + [t for t in order if t not in same_set]
for t in order:
    cands = [s for s in src_pool if s not in used]
    rng.shuffle(cands)
    if t in same_set:
        ptype = "same_label"
        cands.sort(key=lambda s: (-jac(pos[t], pos[s]), contrast(pos[t], pos[s])))
    else:
        ptype = "contrast"
        cands.sort(key=lambda s: -contrast(pos[t], pos[s]))
    s = rng.choice(cands[:a.top_k]); used.add(s)
    T, S = by_sid[t], by_sid[s]
    pairs.append(dict(
        target_study_id=int(t), target_subject_id=int(T["pid"]), target_image_path=T["primary_image_path"],
        source_study_id=int(s), source_subject_id=int(S["pid"]), source_image_path=S["primary_image_path"],
        reference_text=refs[t][0], reference_method=refs[t][1],
        source_reference_text=refs[s][0], source_reference_method=refs[s][1],
        seed=a.seed, pair_type=ptype, target_is_normal=t in normal,
        target_labels=sorted(pos[t]), source_labels=sorted(pos[s]),
        jaccard=round(jac(pos[t], pos[s]), 3), contrast=round(contrast(pos[t], pos[s]), 3)))
pairs.sort(key=lambda p: p["target_study_id"])

# ---- outputs ----
priv = ROOT / f"data/{a.name}_pairs.jsonl"
with open(priv, "w") as f:
    for p in pairs: f.write(json.dumps(p) + "\n")
public = ROOT / f"data/{a.name}_public_manifest.jsonl"
with open(public, "w") as f:
    for p in pairs:
        f.write(json.dumps({k: p[k] for k in ["target_study_id", "source_study_id", "pair_type", "seed"]}) + "\n")

base = a.url_base
if not base and Path(a.url_list).exists():
    first = open(a.url_list).readline().strip()
    base = first[:first.find("files/")] if "files/" in first else None
base = base or "https://physionet.org/files/mimic-cxr-jpg/2.1.0/"
need = []
for p in pairs:
    for rel in [p["target_image_path"], p["source_image_path"]]:
        rel_files = rel[rel.find("files/"):] if "files/" in rel else rel
        if not (Path(a.image_root) / rel_files).exists() and not Path(rel).exists():
            need.append(base + rel_files)
dl = ROOT / f"data/{a.name}_image_urls.txt"
open(dl, "w").write("\n".join(sorted(set(need))) + ("\n" if need else ""))

df = pd.DataFrame(pairs)
summary = dict(
    seed=a.seed, n_pairs=len(pairs), pool_size=len(pool), source_pool=len(src_pool), pool_normal=len(norm_pool), pool_abnormal=len(abn_pool),
    pair_types=df.pair_type.value_counts().to_dict(), normal_targets=int(df.target_is_normal.sum()),
    mean_contrast={k: round(v, 3) for k, v in df.groupby("pair_type").contrast.mean().items()},
    mean_jaccard={k: round(v, 3) for k, v in df.groupby("pair_type").jaccard.mean().items()},
    target_finding_counts=pd.Series([l for ls in df.target_labels for l in ls]).value_counts().to_dict(),
    reference_methods=pd.Series(list(df.reference_method) + list(df.source_reference_method)).value_counts().to_dict(),
    images_to_download=len(set(need)), excluded_pilot_studies=len(excl_s - {"None"}))
json.dump(summary, open(ROOT / f"reports/{a.name}_selection_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print(f"\nprivate pairs: {priv}\npublic manifest: {public}\ndownload list: {dl}")
