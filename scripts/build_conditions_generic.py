"""Build 5-condition rows for any dataset's pairs, copying field layout from the MIMIC scaled conditions (template),
so all runners read new datasets exactly as they read MIMIC. Usage: python build_conditions_generic.py <dataset>"""
import sys, json, collections
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; ds = sys.argv[1]
tmpl = {}
for r in map(json.loads, open(ROOT / "data/project_a_scaled_conditions.jsonl")):
    tmpl.setdefault(r["condition"], r)
c4 = next(v for k, v in tmpl.items() if k.startswith("C4"))
print("MIMIC C4 context format (template):", repr(c4.get("context_text"))[:200])
pairs = [json.loads(l) for l in open(ROOT / f"data/project_a_{ds}_pairs.jsonl")]
out = []
for p in pairs:
    for cond, t in tmpl.items():
        r = dict(t); c = cond[:2]
        for k in ["target_study_id", "target_subject_id", "target_image_path", "source_study_id", "source_subject_id",
                  "source_image_path", "reference_text", "reference_method", "seed"]:
            r[k] = p[k]
        r["image_path"] = p["target_image_path"] if c == "C1" else p["source_image_path"] if c == "C2" else t.get("image_path")
        has_ctx = bool(t.get("context_text"))
        view = p.get("target_view") if p.get("target_view") not in (None, "NA") else "unknown"
        r["context"] = {"view_position": view} if has_ctx else None
        lines = ["The following structured imaging metadata is available before interpreting the image:"]
        if view in ("AP", "PA"):
            lines += [f"View position: {view}", "View: " + {"AP": "antero-posterior", "PA": "postero-anterior"}[view]]
        else:
            lines += ["View position: not recorded"]
        r["context_text"] = "\n".join(lines) if has_ctx else t.get("context_text")
        r["dataset"] = ds; out.append(r)
with open(ROOT / f"data/project_a_{ds}_conditions.jsonl", "w") as f:
    for r in out: f.write(json.dumps(r) + "\n")
print(f"{ds}: {len(out)} rows | {dict(collections.Counter(r['condition'] for r in out))}")
