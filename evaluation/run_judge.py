import json, os
from pathlib import Path
from llama_cpp import Llama
ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "results/project_a_judge"
MODEL = os.environ.get("JUDGE_MODEL", str(ROOT / "models/judge/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"))

SYSTEM = """You are an expert radiologist evaluating a machine-generated chest X-ray description against the radiologist's reference report for the same image. You cannot see the image; judge only against the reference report.

1. output_findings: list each POSITIVE finding in the GENERATED text (abnormalities, devices/lines/tubes, specific anatomical observations). Do not list generic negatives ("no pneumothorax", "lungs are clear") unless the reference contradicts them; if it does, list them as contradicted.
   status: "supported" = reference states or clearly implies it; "contradicted" = reference states the opposite; "not_in_reference" = reference does not mention it.
   clinically_important: true if it would matter for patient management (e.g. pneumothorax, effusion, consolidation, edema, mass, cardiomegaly, device malposition).
2. missed_reference_findings: salient findings in the REFERENCE that the generated text omits (ignore minor/incidental ones).
3. specificity: 0 = generic text that could describe almost any chest X-ray; 1 = some study-specific content; 2 = clearly study-specific (named findings with location/laterality, or devices).
4. reference_agreement: 0 = poor (main findings missed or contradicted); 1 = partial; 2 = substantial (most main findings, minor errors); 3 = strong (main findings, no important errors).
5. rationale: one or two sentences.
Respond only with JSON."""

SCHEMA = {"type": "object", "properties": {
    "output_findings": {"type": "array", "items": {"type": "object", "properties": {
        "finding": {"type": "string"},
        "status": {"type": "string", "enum": ["supported", "contradicted", "not_in_reference"]},
        "clinically_important": {"type": "boolean"}},
        "required": ["finding", "status", "clinically_important"]}},
    "missed_reference_findings": {"type": "array", "items": {"type": "string"}},
    "specificity": {"type": "integer", "enum": [0, 1, 2]},
    "reference_agreement": {"type": "integer", "enum": [0, 1, 2, 3]},
    "rationale": {"type": "string"}},
    "required": ["output_findings", "missed_reference_findings", "specificity", "reference_agreement", "rationale"]}

shard = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
nshard = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1))
items = [json.loads(l) for l in open(J / "items.jsonl")][shard::nshard]
done = set()
for fp in J.glob("judge_shard*.jsonl"):
    for l in open(fp):
        r = json.loads(l)
        if r.get("ok"): done.add(r["item_id"])
todo = [it for it in items if it["item_id"] not in done]
out = J / f"judge_shard{shard}.jsonl"
print(f"shard {shard}/{nshard}: {len(items)} items, {len(todo)} to do", flush=True)
if not todo: raise SystemExit

llm = Llama(model_path=MODEL, n_ctx=4096, n_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)), seed=0, verbose=False)
with open(out, "a") as f:
    for n, it in enumerate(todo, 1):
        msg = f"REFERENCE REPORT:\n{it['reference']}\n\nGENERATED TEXT:\n{it['output']}"
        rec = dict(item_id=it["item_id"], ok=False)
        for attempt in range(2):
            try:
                r = llm.create_chat_completion(
                    messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}],
                    response_format={"type": "json_object", "schema": SCHEMA},
                    temperature=0.0, max_tokens=900)
                rec["raw"] = r["choices"][0]["message"]["content"]
                rec["parsed"] = json.loads(rec["raw"]); rec["ok"] = True; break
            except Exception as e:
                rec["error"] = repr(e)[:300]
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(f"{n}/{len(todo)} ok={rec['ok']}", flush=True)
