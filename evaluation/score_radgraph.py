"""RadGraph F1 (partial: entities + relations) for every judge item, written in judge format to
results/project_a_radgraph_<DS>_<model>/ so evaluation/analyze_scaled.py runs with SCORER=radgraph.
Usage: DS=scaled python evaluation/score_radgraph.py   (DS defaults to scaled = MIMIC Study 1)"""
import os, json, shutil
from pathlib import Path
import torch
torch.set_num_threads(int(os.environ.get("RG_THREADS", 4)))
from radgraph import F1RadGraph
ROOT = Path(__file__).resolve().parents[1]; DS = os.environ.get("DS", "scaled")
f1 = F1RadGraph(reward_level="all", model_type="radgraph-xl")
for tag in ["lingshu", "maira2", "medgemma", "qwen", "chexagent"]:
    J = ROOT / f"results/project_a_judge_{DS}_{tag}"
    if not (J / "items.jsonl").exists(): print(f"{tag}: no items, skipped"); continue
    O = ROOT / f"results/project_a_radgraph_{DS}_{tag}"; O.mkdir(parents=True, exist_ok=True)
    shutil.copy(J / "key.csv", O / "key.csv")
    items = [json.loads(l) for l in open(J / "items.jsonl")]
    with open(O / "judge_shard0.jsonl", "w") as f:
        for i in range(0, len(items), 20):
            chunk = items[i:i + 20]
            _, per, _, _ = f1(hyps=[it["output"] for it in chunk], refs=[it["reference"] for it in chunk])
            simple, partial, complete = per if isinstance(per, (tuple, list)) and len(per) == 3 and isinstance(per[0], (list, tuple)) else (per, per, per)
            for it, s, p, c in zip(chunk, simple, partial, complete):
                f.write(json.dumps(dict(item_id=it["item_id"], ok=True, parsed=dict(
                    reference_agreement=round(float(p), 4), rg_simple=round(float(s), 4), rg_complete=round(float(c), 4)))) + "\n")
            print(f"{tag}: {min(i + 20, len(items))}/{len(items)}", flush=True)
    print(f"{tag}: done -> {O}", flush=True)
