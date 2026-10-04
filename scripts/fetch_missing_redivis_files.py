"""Fetch missing/incomplete files of one Redivis file index, one at a time with a timeout.
Files keep their full relative path (names like patientX/studyY/view1_frontal.png repeat).
Usage: python fetch_missing_redivis_files.py <index_name> [timeout_seconds]"""
import os, sys, multiprocessing as mp
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
DATASET = "chexpert_plus:5yyj"
name = sys.argv[1]; timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 120
import re
safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
lst = pd.read_parquet(ROOT / f"data/chexpert-plus/file_indexes/{safe}.parquet")
out = ROOT / f"data/chexpert-plus/files/{safe}"

def fetch(file_name, dest_dir):
    os.environ["REDIVIS_API_TOKEN"] = (ROOT / ".redivis_token").read_text().strip()
    import redivis
    t = redivis.organization("AIMI").dataset(DATASET).table(name)
    t.file(file_name).download(path=str(dest_dir), overwrite=True)

def missing():
    return [(n, s) for n, s in zip(lst.file_name, lst["size"])
            if not (out / n).exists() or (out / n).stat().st_size != s]

for rnd in range(1, 4):
    todo = missing()
    print(f"round {rnd}: {len(todo)} of {len(lst)} files to fetch", flush=True)
    if not todo: break
    for n, s in todo:
        dest = out / n; dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists(): dest.unlink()
        p = mp.Process(target=fetch, args=(n, dest.parent)); p.start(); p.join(timeout)
        if p.is_alive():
            p.terminate(); p.join(); print(f"  timeout: {n}", flush=True); continue
        ok = dest.exists() and dest.stat().st_size == s
        if not ok:
            stray = dest.parent / Path(n).name
            print(f"  {'ok' if ok else 'FAILED'}: {n}" + ("" if ok else f" (exists={dest.exists()})"), flush=True)
        else:
            print(f"  ok: {n}", flush=True)
left = missing()
print(f"done: {len(lst) - len(left)}/{len(lst)} complete" + (f"; still missing {len(left)}" if left else ""), flush=True)
sys.exit(1 if left else 0)
