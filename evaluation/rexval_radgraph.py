"""ReXVal: RadGraph F1 (simple / partial / complete) for each candidate vs ground truth."""
import json
from pathlib import Path
import pandas as pd, torch
torch.set_num_threads(4)
from radgraph import F1RadGraph
ROOT = Path(__file__).resolve().parents[1]; J = ROOT / "results/rexval_judge"
items = [json.loads(l) for l in open(J / "items.jsonl")]
f1 = F1RadGraph(reward_level="all", model_type="radgraph-xl")
rows = []
for i in range(0, len(items), 20):
    chunk = items[i:i + 20]
    _, per, _, _ = f1(hyps=[it["output"] for it in chunk], refs=[it["reference"] for it in chunk])
    simple, partial, complete = per if isinstance(per, (tuple, list)) and len(per) == 3 and isinstance(per[0], (list, tuple)) else (per, per, per)
    for it, s, p, c in zip(chunk, simple, partial, complete):
        rows.append(dict(item_id=it["item_id"], radgraph=float(p), radgraph_simple=float(s), radgraph_complete=float(c)))
    print(f"{min(i + 20, len(items))}/{len(items)}", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "results/rexval_radgraph.csv", index=False)
print(f"scored {len(rows)} candidates with RadGraph")
