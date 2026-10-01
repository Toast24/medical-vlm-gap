import json, os, re, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = Path(__file__).resolve().parents[1]
J = Path(os.environ.get("JUDGE_DIR", ROOT / "results/project_a_judge"))
MDIR, MFILE = str(ROOT / "models/judge"), "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
SYSTEM = re.search(r'SYSTEM = """(.*?)"""', open(ROOT / "evaluation/run_judge.py").read(), re.S).group(1)
SYSTEM += ('\n\nOutput exactly one JSON object with this structure and nothing else:\n'
           '{"output_findings": [{"finding": "...", "status": "supported|contradicted|not_in_reference", "clinically_important": true}], '
           '"missed_reference_findings": ["..."], "specificity": 0, "reference_agreement": 0, "rationale": "..."}')
BATCH = int(os.environ.get("JUDGE_BATCH", 8))

def prompt(it):
    user = f"REFERENCE REPORT:\n{it['reference']}\n\nGENERATED TEXT:\n{it['output']}"
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + SYSTEM + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + user + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")

def parse(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m: return None
    try: p = json.loads(m.group(0))
    except Exception: return None
    try: p["specificity"], p["reference_agreement"] = int(p["specificity"]), int(p["reference_agreement"])
    except Exception: return None
    if p["specificity"] not in (0, 1, 2) or p["reference_agreement"] not in (0, 1, 2, 3): return None
    if not isinstance(p.get("output_findings"), list) or not isinstance(p.get("missed_reference_findings"), list): return None
    for f in p["output_findings"]:
        if not isinstance(f, dict) or f.get("status") not in ("supported", "contradicted", "not_in_reference"): return None
    return p

items = [json.loads(l) for l in open(J / "items.jsonl")]
done = set()
for fp in J.glob("judge_shard*.jsonl"):
    for l in open(fp):
        r = json.loads(l)
        if r.get("ok"): done.add(r["item_id"])
todo = [it for it in items if it["item_id"] not in done]
print(f"{len(items)} items, {len(todo)} to do", flush=True)
if not todo: raise SystemExit

tok = AutoTokenizer.from_pretrained(MDIR, gguf_file=MFILE)
tok.padding_side = "left"
if tok.pad_token is None: tok.pad_token = tok.eos_token
eot = tok.convert_tokens_to_ids("<|eot_id|>")
print("loading model (dequantizing GGUF, takes a few minutes)...", flush=True)
model = AutoModelForCausalLM.from_pretrained(MDIR, gguf_file=MFILE, torch_dtype=torch.bfloat16).to("cuda").eval()

def gen(batch, sample=False):
    enc = tok([prompt(i) for i in batch], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    kw = dict(do_sample=True, temperature=0.3, top_p=0.95) if sample else dict(do_sample=False)
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=900, eos_token_id=[eot, tok.eos_token_id], pad_token_id=tok.pad_token_id, **kw)
    return tok.batch_decode(o[:, enc.input_ids.shape[1]:], skip_special_tokens=True)

with open(J / "judge_shard0.jsonl", "a") as f:
    for b in range(0, len(todo), BATCH):
        batch = todo[b:b + BATCH]
        for it, txt in zip(batch, gen(batch)):
            p = parse(txt)
            if p is None:
                for _ in range(2):
                    txt = gen([it], sample=True)[0]; p = parse(txt)
                    if p: break
            f.write(json.dumps(dict(item_id=it["item_id"], ok=p is not None, raw=txt, parsed=p)) + "\n"); f.flush()
        print(f"{min(b + BATCH, len(todo))}/{len(todo)} done", flush=True)
