import json, re, zipfile, collections
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
MIMIC = ROOT / "data/mimic-cxr-jpg"
OUT = ROOT / "results/project_a_source_references.jsonl"
norm = lambda x: str(x).strip().lstrip("sS").split(".")[0]

cond = [json.loads(l) for l in open(ROOT / "data/project_a_pilot_conditions.jsonl") if l.strip()]
sources = {norm(r["source_study_id"]): str(r.get("source_subject_id", "")).lstrip("pP").split(".")[0]
           for r in cond if r.get("source_study_id")}
print(f"source studies: {len(sources)}")

man = pd.read_csv(MIMIC / "mimic_cxr_test_manifest.csv", dtype=str)
print("manifest columns:", list(man.columns))
scol = next((c for c in ["study_id", "target_study_id", "study"] if c in man.columns), None)
ref_by = {}
if scol and "reference_text" in man.columns:
    for _, r in man.iterrows():
        if isinstance(r["reference_text"], str) and r["reference_text"].strip():
            ref_by.setdefault(norm(r[scol]), (r["reference_text"], r.get("reference_method", "manifest")))

def extract(txt):
    sec = {}
    for name in ["FINDINGS", "IMPRESSION"]:
        m = re.search(rf"{name}:\s*(.*?)(?=\n\s*[A-Z][A-Z /()]+:|\Z)", txt, re.S)
        if m and m.group(1).strip(): sec[name] = " ".join(m.group(1).split())
    if "FINDINGS" in sec and "IMPRESSION" in sec:
        return f"FINDINGS: {sec['FINDINGS']} IMPRESSION: {sec['IMPRESSION']}", "findings+impression"
    if "IMPRESSION" in sec: return sec["IMPRESSION"], "impression"
    if "FINDINGS" in sec: return sec["FINDINGS"], "findings"
    return " ".join(txt.split()), "narrative_fallback"

zf = zipfile.ZipFile(MIMIC / "mimic-cxr-reports.zip") if (MIMIC / "mimic-cxr-reports.zip").exists() else None
zip_index = {}
if zf:
    for n in zf.namelist():
        m = re.search(r"/s(\d+)\.txt$", n)
        if m: zip_index[m.group(1)] = n

found = collections.Counter(); methods = collections.Counter()
with open(OUT, "w") as f:
    for s, subj in sources.items():
        text = method = None
        if s in ref_by:
            text, method = ref_by[s]; found["manifest"] += 1
        else:
            p = MIMIC / "files" / f"p{subj[:2]}" / f"p{subj}" / f"s{s}.txt"
            raw = p.read_text() if subj and p.exists() else (zf.read(zip_index[s]).decode() if s in zip_index else None)
            if raw:
                text, method = extract(raw); found["raw_report"] += 1
            else:
                found["MISSING"] += 1; continue
        methods[method] += 1
        f.write(json.dumps(dict(source_study_id=s, reference_text=text, reference_method=method)) + "\n")
print("found via:", dict(found)); print("methods:", dict(methods)); print(f"wrote {OUT}")
