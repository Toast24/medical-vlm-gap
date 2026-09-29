#!/usr/bin/env python3

import os
import json
import time

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq


PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.dirname(os.path.abspath(__file__)),
)

MODEL_PATH = os.environ.get(
    "MAIRA2_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "maira2"),
)

MANIFEST = os.environ.get(
    "MIMIC_MANIFEST",
    os.path.join(
        PROJECT_ROOT,
        "data",
        "mimic-cxr-jpg",
        "mimic_cxr_test_manifest.csv",
    ),
)

OUTPUT = os.environ.get(
    "MAIRA2_OUTPUT",
    os.path.join(
        PROJECT_ROOT,
        "results",
        "maira2_mimic_test.jsonl",
    ),
)


def load_completed():
    """Load study IDs already successfully processed."""
    completed = set()

    if not os.path.exists(OUTPUT):
        return completed

    with open(OUTPUT, "r") as f:
        for line in f:
            try:
                record = json.loads(line)
                if record.get("status") == "success":
                    completed.add(str(record["study_id"]))
            except Exception:
                pass

    return completed


print("=== MAIRA-2 BATCH INFERENCE ===")
print("Job ID:", os.environ.get("SLURM_JOB_ID", "local"))
print("Node:", os.environ.get("HOSTNAME", "unknown"))
print("CUDA:", os.environ.get("CUDA_VISIBLE_DEVICES", "unknown"))

print("\nLoading manifest...")
df = pd.read_csv(MANIFEST)

print("Total studies:", len(df))

completed = load_completed()

print("Already completed:", len(completed))
print("Remaining:", len(df) - len(completed))

print("\nLoading MAIRA-2...")

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()

print("Model loaded!")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "Allocated VRAM:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("\nStarting inference...")
print("Output:", OUTPUT)

with open(OUTPUT, "a") as fout:

    for idx, row in df.iterrows():

        study_id = str(row["study_id"])

        if study_id in completed:
            continue

        image_path = os.path.join(
    os.path.join(
        PROJECT_ROOT,
        "data",
        "mimic-cxr-jpg",
    ),
    row["primary_image_path"]
)

        print(
            f"\n[{idx + 1}/{len(df)}] "
            f"Study {study_id} | "
            f"View {row['primary_view']}"
        )

        start_time = time.time()

        try:

            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            image = Image.open(image_path).convert("RGB")

            # Image-only protocol.
            # No indication, technique, comparison,
            # prior report, or ground-truth text is supplied.
            inputs = processor.format_and_preprocess_reporting_input(
                current_frontal=image,
                current_lateral=None,
                prior_frontal=None,
                indication=None,
                technique=None,
                comparison=None,
                prior_report=None,
                get_grounding=False,
                return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():

                output = model.generate(
                    **inputs,
                    max_new_tokens=100
                )

            # Decode only newly generated tokens.
            input_length = inputs["input_ids"].shape[1]

            generated_tokens = output[:, input_length:]

            prediction = processor.decode(
                generated_tokens[0],
                skip_special_tokens=True
            ).strip()

            elapsed = time.time() - start_time

            record = {
                "subject_id": int(row["subject_id"]),
                "study_id": int(row["study_id"]),
                "primary_dicom_id": row["primary_dicom_id"],
                "view": row["primary_view"],
                "image_path": image_path,
                "report_path": row["report_path"],
                "prediction": prediction,
                "inference_time_sec": round(elapsed, 3),
                "status": "success"
            }

            fout.write(json.dumps(record) + "\n")
            fout.flush()

            completed.add(study_id)

            print("Prediction:", prediction)
            print("Time:", round(elapsed, 2), "sec")

        except Exception as e:

            elapsed = time.time() - start_time

            record = {
                "subject_id": int(row["subject_id"]),
                "study_id": int(row["study_id"]),
                "primary_dicom_id": row["primary_dicom_id"],
                "view": row["primary_view"],
                "image_path": image_path,
                "report_path": row["report_path"],
                "prediction": "",
                "inference_time_sec": round(elapsed, 3),
                "status": "error",
                "error": repr(e)
            }

            fout.write(json.dumps(record) + "\n")
            fout.flush()

            print("ERROR:", repr(e))


print("\n=== BATCH COMPLETE ===")

success_count = 0
error_count = 0

if os.path.exists(OUTPUT):
    with open(OUTPUT, "r") as f:
        for line in f:
            try:
                record = json.loads(line)

                if record.get("status") == "success":
                    success_count += 1
                elif record.get("status") == "error":
                    error_count += 1

            except Exception:
                pass

print("Successful:", success_count)
print("Errors:", error_count)
print("Expected:", len(df))
print("Output:", OUTPUT)
