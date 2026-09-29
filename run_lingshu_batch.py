#!/usr/bin/env python3

import os
import json
import time

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.dirname(os.path.abspath(__file__)),
)

MODEL_PATH = os.environ.get(
    "LINGSHU_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "lingshu"),
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
    "LINGSHU_OUTPUT",
    os.path.join(
        PROJECT_ROOT,
        "results",
        "lingshu_mimic_test.jsonl",
    ),
)

DATA_ROOT = os.environ.get(
    "MIMIC_CXR_ROOT",
    os.path.join(PROJECT_ROOT, "data", "mimic-cxr-jpg"),
)


os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("=== LINGSHU MIMIC-CXR BATCH INFERENCE ===")
print("Job:", os.environ.get("SLURM_JOB_ID", "local"))
print("Node:", os.uname().nodename)


# --------------------------------------------------
# Load model
# --------------------------------------------------

print("\nLoading Lingshu-7B...")

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    max_memory={0: "45GiB"},
)

model.eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Model loaded!")
print("GPU:", torch.cuda.get_device_name(0))
print(
    "VRAM allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)


# --------------------------------------------------
# Load manifest
# --------------------------------------------------

df = pd.read_csv(MANIFEST)

print("\nStudies:", len(df))


# --------------------------------------------------
# Resume support
# Only successful studies are considered completed
# --------------------------------------------------

completed = set()

if os.path.exists(OUTPUT):
    print("Existing output found. Checking completed studies...")

    with open(OUTPUT, "r") as f:
        for line in f:
            try:
                record = json.loads(line)

                if record.get("status") == "success":
                    completed.add(str(record["study_id"]))

            except Exception:
                pass

print("Already completed:", len(completed))
print("Remaining:", len(df) - len(completed))


# --------------------------------------------------
# Inference
# --------------------------------------------------

prompt = (
    "Describe this chest X-ray briefly. "
    "Focus on the main radiographic findings and abnormalities. "
    "Do not speculate beyond what is visible."
)

start_all = time.time()

with open(OUTPUT, "a") as fout:

    for idx, row in df.iterrows():

        study_id = str(row["study_id"])

        if study_id in completed:
            continue

        image_path = os.path.join(
            DATA_ROOT,
            row["primary_image_path"]
        )

        print(
            f"\n[{idx + 1}/{len(df)}] "
            f"Study {study_id} | {row['primary_view']}"
        )

        t0 = time.time()

        try:

            if not os.path.isfile(image_path):
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            image = Image.open(image_path).convert("RGB")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt"
            ).to(model.device)

            with torch.inference_mode():

                output = model.generate(
                    **inputs,
                    max_new_tokens=128
                )

            input_length = inputs.input_ids.shape[1]

            generated_tokens = output[
                0,
                input_length:
            ]

            response = processor.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            elapsed = time.time() - t0

            record = {
                "subject_id": int(row["subject_id"]),
                "study_id": int(row["study_id"]),
                "primary_dicom_id": row["primary_dicom_id"],
                "view": row["primary_view"],
                "image_path": image_path,
                "report_path": row["report_path"],
                "prompt": prompt,
                "prediction": response,
                "inference_time_sec": round(elapsed, 2),
                "status": "success"
            }

            fout.write(json.dumps(record) + "\n")
            fout.flush()

            completed.add(study_id)

            print("Prediction:", response)
            print("Time:", round(elapsed, 2), "sec")

        except Exception as e:

            elapsed = time.time() - t0

            print("ERROR:", repr(e))

            record = {
                "subject_id": int(row["subject_id"]),
                "study_id": int(row["study_id"]),
                "primary_dicom_id": row["primary_dicom_id"],
                "view": row["primary_view"],
                "image_path": image_path,
                "report_path": row["report_path"],
                "prediction": None,
                "inference_time_sec": round(elapsed, 2),
                "status": "error",
                "error": repr(e)
            }

            fout.write(json.dumps(record) + "\n")
            fout.flush()


print("\n=== BATCH COMPLETE ===")

elapsed_all = time.time() - start_all

print(
    "Total runtime:",
    round(elapsed_all / 3600, 2),
    "hours"
)

print("Output:", OUTPUT)
