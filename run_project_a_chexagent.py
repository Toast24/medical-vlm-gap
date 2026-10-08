"""Project A runner for StanfordAIMI/CheXagent-2-3b (custom code, model-card chat format).
C1/C2 supplied image, C3 512x512 black PNG, C4/C5 text-only. Greedy decoding.
Env: PROJECT_A_CONDITIONS, PROJECT_A_OUTPUT, PROJECT_A_N, CHEXAGENT_MODEL_PATH, MAX_NEW_TOKENS, SMOKE=1."""
import os, json, collections, sys, torch, numpy as np
from PIL import Image
ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("CHEXAGENT_MODEL_PATH", f"{ROOT}/models/chexagent-2-3b")
COND = os.environ["PROJECT_A_CONDITIONS"]; OUT = os.environ["PROJECT_A_OUTPUT"]
N = int(os.environ.get("PROJECT_A_N", 20)); MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", 384))
SMOKE = os.environ.get("SMOKE") == "1"; SYSTEM = "You are a helpful assistant."

rows = [json.loads(l) for l in open(COND) if l.strip()]
counts = collections.Counter(r["condition"] for r in rows); print(f"{len(rows)} records: {dict(counts)}")
if not SMOKE and (len(rows) != 5 * N or set(counts.values()) != {N}): raise RuntimeError(f"unexpected counts {dict(counts)}")
if SMOKE:
    first = rows[0]["target_study_id"]; rows = [r for r in rows if r["target_study_id"] == first]
    OUT = OUT.replace(".jsonl", "_smoke.jsonl"); print(f"SMOKE: {len(rows)} rows -> {OUT}")

BLANK = os.path.join(ROOT, "results/_blank_512x512_black.png")
if not os.path.exists(BLANK): Image.fromarray(np.zeros((512, 512), dtype=np.uint8)).convert("RGB").save(BLANK)
def resolve(p):
    for c in [p, os.path.join(ROOT, "data/mimic-cxr-jpg", p), os.path.join(ROOT, p)]:
        if p and os.path.exists(c): return c
    raise FileNotFoundError(p)
def build(r):
    c, prompt = r["condition"], r["prompt"]
    if c.startswith(("C1", "C2")): return resolve(r["image_path"]), prompt
    if c.startswith("C3"): return BLANK, prompt
    if c.startswith("C4"):
        ctx = r.get("context_text") or ""
        if not ctx: raise RuntimeError(f"C4 has no context_text for {r['target_study_id']}")
        return None, ctx + "\n\n" + prompt
    return None, prompt

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        x = json.loads(l)
        if x.get("prediction"): done.add((str(x["target_study_id"]), x["condition"]))
todo = [r for r in rows if (str(r["target_study_id"]), r["condition"]) not in done]
print(f"{len(todo)} to do", flush=True)
if not todo: sys.exit(0)

from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers; print("transformers", transformers.__version__, flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto", trust_remote_code=True).to(torch.bfloat16).eval()

def generate(img, text):
    parts = ([{"image": img}] if img else []) + [{"text": text}]
    conv = [{"from": "system", "value": SYSTEM}, {"from": "human", "value": tok.from_list_format(parts)}]
    ids = tok.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(ids, attention_mask=torch.ones_like(ids), pad_token_id=tok.eos_token_id,
                             do_sample=False, num_beams=1, max_new_tokens=MAX_NEW, use_cache=True)[0]
    new = out[ids.size(1):]
    return tok.decode(new, skip_special_tokens=True).strip(), int(new.size(0))

fails = 0
with open(OUT, "a") as f:
    for i, r in enumerate(todo, 1):
        rec = dict(target_study_id=r["target_study_id"], source_study_id=r.get("source_study_id"), condition=r["condition"],
                   model="CheXagent-2-3b", precision="bf16", system_prompt=SYSTEM, max_new_tokens=MAX_NEW)
        try:
            img, text = build(r); rec["image_used"] = img
            rec["prediction"], rec["new_tokens"] = generate(img, text); rec["hit_token_limit"] = rec["new_tokens"] >= MAX_NEW
            if not rec["prediction"]: raise RuntimeError("empty output")
        except Exception as e:
            rec["prediction"], rec["error"] = None, repr(e)[:300]; fails += 1
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"{i}/{len(todo)} {r['condition']} ok={rec['prediction'] is not None} tokens={rec.get('new_tokens')}", flush=True)
if SMOKE and fails: sys.exit(1)
