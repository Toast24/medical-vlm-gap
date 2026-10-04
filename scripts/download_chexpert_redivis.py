"""Download CheXpert Plus from Redivis: all data tables, every file-index listing, and the files of
small indexes (labels, validation PNGs, RadGraph). Training images are NOT downloaded here; they are
fetched selectively after pair selection. Resumable. Token read from .redivis_token (git-ignored)."""
import os, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ["REDIVIS_API_TOKEN"] = (ROOT / ".redivis_token").read_text().strip()
import redivis

DATASET_REF = os.environ.get("REDIVIS_DATASET", "chexpert_plus:5yyj")
DOWNLOAD_FILES = {"CheXpert Labels", "PNG_valid"}
MAX_FILE_BYTES = 5e9
BASE = ROOT / "data/chexpert-plus"
safe = lambda s: re.sub(r"[^A-Za-z0-9_.-]+", "_", s)

ds = redivis.organization(os.environ.get("REDIVIS_ORG", "AIMI")).dataset(DATASET_REF)
for t in ds.list_tables():
    try: t.get()
    except Exception: pass
    p = getattr(t, "properties", {}) or {}
    name = safe(t.name)
    if not p.get("isFileIndex"):
        dest = BASE / "tables" / f"{name}.parquet"; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): print(f"{t.name}: table already downloaded", flush=True); continue
        print(f"{t.name}: downloading table ({p.get('numRows')} rows)...", flush=True)
        df = t.to_pandas_dataframe(); tmp = dest.with_suffix(".part"); df.to_parquet(tmp); tmp.rename(dest)
        print(f"{t.name}: saved ({len(df)} rows)", flush=True); continue

    idx = BASE / "file_indexes" / f"{name}.parquet"; idx.parent.mkdir(parents=True, exist_ok=True)
    if not idx.exists():
        listing = t.to_pandas_dataframe(); listing.to_parquet(idx)
    else:
        import pandas as pd; listing = pd.read_parquet(idx)
    total = float(listing["size"].sum()) if "size" in listing else float("nan")
    print(f"{t.name}: file index saved ({len(listing)} files, {total / 1e9:.2f} GB)", flush=True)
    if t.name not in DOWNLOAD_FILES: continue
    if not total < MAX_FILE_BYTES: print(f"{t.name}: over {MAX_FILE_BYTES / 1e9:.0f} GB, files NOT downloaded", flush=True); continue
    out = BASE / "files" / name; out.mkdir(parents=True, exist_ok=True)
    have = {f.name for f in out.rglob("*") if f.is_file()}
    if len(have) >= len(listing): print(f"{t.name}: files already downloaded", flush=True); continue
    print(f"{t.name}: downloading {len(listing)} files...", flush=True)
    try:
        t.download_files(path=str(out))
    except (AttributeError, TypeError):
        for fid in listing["file_id"]:
            redivis.File(fid).download(path=str(out))
    print(f"{t.name}: files done ({sum(1 for f in out.rglob('*') if f.is_file())} on disk)", flush=True)
import sys, pandas as pd
missing = []
for n in DOWNLOAD_FILES:
    idx = BASE / "file_indexes" / f"{safe(n)}.parquet"
    if not idx.exists(): missing.append(f"{n}: no listing"); continue
    lst = pd.read_parquet(idx)
    if float(lst["size"].sum()) >= MAX_FILE_BYTES: continue
    have = sum(1 for f in (BASE / "files" / safe(n)).rglob("*") if f.is_file())
    if have < len(lst): missing.append(f"{n}: {have}/{len(lst)} files")
if not (BASE / "tables" / "df_chexpert_plus_240401.parquet").exists(): missing.append("main table missing")
if missing: print("INCOMPLETE:", "; ".join(missing), flush=True); sys.exit(1)
print("CheXpert Plus download finished", flush=True)
