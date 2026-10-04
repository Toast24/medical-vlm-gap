"""IU X-ray (OpenI) -> manifest.csv: one frontal image + FINDINGS/IMPRESSION reference per report.
Frontal view is not labelled in the XML: heuristic = image id ending '-1001', else first listed (recorded)."""
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; IU = ROOT / "data/iu-xray"
pngs = {p.stem: p for p in IU.rglob("*.png")}
rows, skipped = [], {"no_image": 0, "no_text": 0}
for x in sorted(IU.rglob("*.xml")):
    t = ET.parse(x).getroot()
    sec = {}
    for a in t.iter("AbstractText"):
        txt = " ".join((a.text or "").split())
        if txt: sec[(a.get("Label") or "").upper()] = txt
    imgs = [p.get("id") for p in t.iter("parentImage") if p.get("id") in pngs]
    if not imgs: skipped["no_image"] += 1; continue
    f, i = sec.get("FINDINGS"), sec.get("IMPRESSION")
    if f and i: ref, m = f"FINDINGS: {f} IMPRESSION: {i}", "findings+impression"
    elif i: ref, m = i, "impression"
    elif f: ref, m = f, "findings"
    else: skipped["no_text"] += 1; continue
    front = next((k for k in imgs if k.endswith("-1001")), None)
    sid = 90_000_000 + int(x.stem)
    rows.append(dict(study_id=sid, subject_id=sid, image_path=str(pngs[front or imgs[0]]), view="NA",
                     view_source="suffix_-1001" if front else "first_listed", n_images=len(imgs),
                     reference_text=ref, reference_method=m, split="all"))
df = pd.DataFrame(rows); df.to_csv(IU / "manifest.csv", index=False)
print(f"IU manifest: {len(df)} studies | skipped {skipped} | view source {df.view_source.value_counts().to_dict()} | methods {df.reference_method.value_counts().to_dict()}")
