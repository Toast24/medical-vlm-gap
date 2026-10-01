import os, json, torch, numpy as np
from PIL import Image
ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("MAIRA2_MODEL_PATH", f"{ROOT}/models/maira2")
COND = os.environ.get("PROJECT_A_CONDITIONS", f"{ROOT}/data/project_a_pilot_conditions.jsonl")
OUT = os.environ.get("PROJECT_A_OUTPUT", f"{ROOT}/results/project_a_maira2_pilot.jsonl")
DRY = os.environ.get("DRY_RUN") == "1"

rows = [json.loads(l) for l in open(COND) if l.strip()]
print(f"{len(rows)} condition records")

def resolve(p):
    for c in [p, os.path.join(ROOT, "data/mimic-cxr-jpg", p), os.path.join(ROOT, p)]:
        if p and os.path.exists(c): return c
    raise FileNotFoundError(p)

blank = Image.fromarray(np.zeros((512, 512), dtype=np.uint8)).convert("RGB")
def plan(r):
    c = r["condition"]
    if c.startswith(("C1", "C2")): return dict(image=resolve(r["image_path"]), technique=None)
    if c.startswith("C3"): return dict(image="BLANK", technique=None)
    if c.startswith("C4"): return dict(image=None, technique=r.get("context_text") or r.get("context"))
    return dict(image=None, technique=None)

plans = [plan(r) for r in rows]
print("all image paths resolve; technique text present for C4:",
      sum(1 for r, p in zip(rows, plans) if r["condition"].startswith("C4") and p["technique"]), "/ 20")
if DRY: raise SystemExit("dry run OK")

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        r = json.loads(l)
        if r.get("prediction"): done.add((str(r["target_study_id"]), r["condition"]))

from transformers import AutoModelForCausalLM, AutoProcessor
import transformers; print("transformers", transformers.__version__)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True).eval().to("cuda")
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

def run(img, technique):
    inp = processor.format_and_preprocess_reporting_input(
        current_frontal=img, current_lateral=None, prior_frontal=None, indication=None,
        technique=technique, comparison=None, prior_report=None, return_tensors="pt", get_grounding=False).to("cuda")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=300, do_sample=False, use_cache=True)
    txt = processor.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True).lstrip()
    return processor.convert_output_to_plaintext_or_grounded_sequence(txt)

with open(OUT, "a") as f:
    for n, (r, p) in enumerate(zip(rows, plans), 1):
        k = (str(r["target_study_id"]), r["condition"])
        if k in done: continue
        rec = dict(target_study_id=r["target_study_id"], source_study_id=r.get("source_study_id"), condition=r["condition"],
                   model="MAIRA-2", technique_used=p["technique"], image_used=p["image"], image_substituted=False)
        try:
            img = blank if p["image"] == "BLANK" else (Image.open(p["image"]).convert("RGB") if p["image"] else None)
            try:
                rec["prediction"] = run(img, p["technique"])
            except Exception as e:
                if img is not None: raise
                rec["image_substituted"] = True; rec["image_used"] = "BLANK (processor requires image)"
                rec["substitution_reason"] = repr(e)[:200]
                rec["prediction"] = run(blank, p["technique"])
        except Exception as e:
            rec["prediction"] = None; rec["error"] = repr(e)[:300]
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"{n}/{len(rows)} {r['condition']} ok={rec['prediction'] is not None} substituted={rec['image_substituted']}", flush=True)
