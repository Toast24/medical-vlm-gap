"""CheXpert Plus validation split -> manifest_valid.csv: one frontal PNG + findings/impression reference per study."""
import re
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; CX = ROOT / "data/chexpert-plus"
d = pd.read_parquet(CX / "tables/df_chexpert_plus_240401.parquet",
                    columns=["path_to_image", "frontal_lateral", "ap_pa", "deid_patient_id", "section_findings", "section_impression"])
v = d[d.path_to_image.str.startswith("valid") & (d.frontal_lateral == "Frontal")].sort_values("path_to_image")
rows, skipped = [], {"no_png": 0, "no_text": 0, "extra_frontal": 0}
seen = set()
for r in v.itertuples():
    m = re.match(r"valid/patient(\d+)/study(\d+)/", r.path_to_image)
    pat, st = int(m.group(1)), int(m.group(2)); sid = 80_000_000 + pat * 10 + st
    if sid in seen: skipped["extra_frontal"] += 1; continue
    png = CX / "files/PNG_valid" / re.sub(r"\.jpg$", ".png", r.path_to_image[len("valid/"):])
    if not png.exists(): skipped["no_png"] += 1; continue
    f = r.section_findings if isinstance(r.section_findings, str) and r.section_findings.strip() else None
    i = r.section_impression if isinstance(r.section_impression, str) and r.section_impression.strip() else None
    f = " ".join(f.split()) if f else None; i = " ".join(i.split()) if i else None
    if f and i: ref, mth = f"FINDINGS: {f} IMPRESSION: {i}", "findings+impression"
    elif i: ref, mth = i, "impression"
    elif f: ref, mth = f, "findings"
    else: skipped["no_text"] += 1; continue
    seen.add(sid)
    rows.append(dict(study_id=sid, subject_id=80_000_000 + pat, image_path=str(png),
                     view=r.ap_pa if isinstance(r.ap_pa, str) else "NA", view_source="metadata",
                     n_images=1, reference_text=ref, reference_method=mth, split="valid"))
df = pd.DataFrame(rows); df.to_csv(CX / "manifest_valid.csv", index=False)
print(f"CheXpert manifest: {len(df)} studies, {df.subject_id.nunique()} patients | skipped {skipped} | views {df.view.value_counts().to_dict()} | methods {df.reference_method.value_counts().to_dict()}")
