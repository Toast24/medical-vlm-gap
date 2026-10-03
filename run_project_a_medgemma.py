"""Project A runner for MedGemma-27B (multimodal), 8-bit by default.

Same conditions file, per-row prompt, greedy decoding and resume behaviour as the other runners.
C1/C2 use the supplied image, C3 a 512x512 black image, C4/C5 run text-only (no image).
Env: PROJECT_A_CONDITIONS, PROJECT_A_OUTPUT, PROJECT_A_N, MEDGEMMA_MODEL_PATH,
     MEDGEMMA_PRECISION (int8 | bf16), MAX_NEW_TOKENS, SMOKE=1 (first study's 5 conditions only).
"""
import os, json, collections, torch, numpy as np
from PIL import Image

ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("MEDGEMMA_MODEL_PATH", f"{ROOT}/models/medgemma-27b-it")
COND = os.environ.get("PROJECT_A_CONDITIONS", f"{ROOT}/data/project_a_pilot_conditions.jsonl")
OUT = os.environ.get("PROJECT_A_OUTPUT", f"{ROOT}/results/project_a_medgemma_pilot.jsonl")
N_EXPECTED = int(os.environ.get("PROJECT_A_N", 20))
PRECISION = os.environ.get("MEDGEMMA_PRECISION", "int8")
MAX_NEW = int(os.environ.get("MAX_NEW_TOKENS", 256))
SMOKE = os.environ.get("SMOKE") == "1"

rows = [json.loads(l) for l in open(COND) if l.strip()]
counts = collections.Counter(r["condition"] for r in rows)
print(f"{len(rows)} condition records: {dict(counts)}")
if not SMOKE and (len(rows) != 5 * N_EXPECTED or set(counts.values()) != {N_EXPECTED}):
    raise RuntimeError(f"expected {N_EXPECTED} studies x 5 conditions, got {dict(counts)}")
if SMOKE:
    first = rows[0]["target_study_id"]
    rows = [r for r in rows if r["target_study_id"] == first]
    OUT = OUT.replace(".jsonl", "_smoke.jsonl")
    print(f"SMOKE: {len(rows)} rows for study {first} -> {OUT}")

def resolve(p):
    for c in [p, os.path.join(ROOT, "data/mimic-cxr-jpg", p), os.path.join(ROOT, p)]:
        if p and os.path.exists(c): return c
    raise FileNotFoundError(p)

BLANK = Image.fromarray(np.zeros((512, 512), dtype=np.uint8)).convert("RGB")
def build(r):
    c, prompt = r["condition"], r["prompt"]
    if c.startswith(("C1", "C2")): return Image.open(resolve(r["image_path"])).convert("RGB"), prompt, r["image_path"]
    if c.startswith("C3"): return BLANK, prompt, "BLANK_512_BLACK"
    if c.startswith("C4"):
        ctx = r.get("context_text") or ""
        if not ctx: raise RuntimeError(f"C4 has no context_text for study {r['target_study_id']}")
        return None, ctx + "\n\n" + prompt, None
    return None, prompt, None

done = set()
if os.path.exists(OUT):
    for l in open(OUT):
        x = json.loads(l)
        if x.get("prediction"): done.add((str(x["target_study_id"]), x["condition"]))
todo = [r for r in rows if (str(r["target_study_id"]), r["condition"]) not in done]
print(f"{len(todo)} to do, {len(done)} already done", flush=True)
if not todo: raise SystemExit

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import transformers; print("transformers", transformers.__version__, "| precision", PRECISION, flush=True)
kw = dict(torch_dtype=torch.bfloat16, device_map="auto")
if PRECISION == "int8":
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_8bit=True, llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"])
model = AutoModelForImageTextToText.from_pretrained(MODEL, **kw).eval()
processor = AutoProcessor.from_pretrained(MODEL)
print("processor:", type(processor).__name__, "| image processor:", type(processor.image_processor).__name__,
      "| tokenizer:", type(processor.tokenizer).__name__, flush=True)
print(f"model loaded; GPU memory {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)

def generate(img, text):
    content = ([{"type": "image", "image": img}] if img is not None else []) + [{"type": "text", "text": text}]
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                           tokenize=True, return_dict=True, return_tensors="pt")
    inputs = inputs.to(model.device, dtype=torch.bfloat16)
    n = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW, do_sample=False, top_p=None, top_k=None)
    return processor.decode(out[0][n:], skip_special_tokens=True).strip(), int(out.shape[-1] - n)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "a") as f:
    for i, r in enumerate(todo, 1):
        rec = dict(target_study_id=r["target_study_id"], source_study_id=r.get("source_study_id"),
                   condition=r["condition"], model="MedGemma-27B-it", precision=PRECISION,
                   max_new_tokens=MAX_NEW)
        try:
            img, text, used = build(r)
            rec["image_used"], rec["text_input_chars"] = used, len(text)
            rec["prediction"], rec["new_tokens"] = generate(img, text)
            rec["hit_token_limit"] = rec["new_tokens"] >= MAX_NEW
        except Exception as e:
            rec["prediction"], rec["error"] = None, repr(e)[:300]
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"{i}/{len(todo)} {r['condition']} ok={rec['prediction'] is not None} "
              f"tokens={rec.get('new_tokens')}", flush=True)
