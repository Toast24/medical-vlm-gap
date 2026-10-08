#!/usr/bin/env python3

import os
import json
import time

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ==============================================================
# Paths
# ==============================================================

PROJECT_ROOT = os.environ.get(
    "PROJECT_ROOT",
    str(os.path.dirname(os.path.abspath(__file__))),
)

MODEL_PATH = os.environ.get(
    "LINGSHU_MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "lingshu"),
)

CONDITIONS = os.environ.get(
    "PROJECT_A_CONDITIONS",
    os.path.join(PROJECT_ROOT, "data", "project_a_pilot_conditions.jsonl"),
)

OUTPUT = os.environ.get(
    "PROJECT_A_OUTPUT",
    os.path.join(PROJECT_ROOT, "results", "project_a_lingshu_pilot.jsonl"),
)


# ==============================================================
# Constants
# ==============================================================

PROMPT = (
    "Describe this chest X-ray briefly. "
    "Focus on the main radiographic findings and abnormalities. "
    "Do not speculate beyond what is visible."
)

PROMPT = os.environ.get("PROJECT_A_PROMPT", PROMPT)
print("prompt:", repr(PROMPT), flush=True)

BLANK_SIZE = (512, 512)
BLANK_VALUE = 0


# ==============================================================
# Setup
# ==============================================================

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

print("=== PROJECT A — LINGSHU PILOT ===")
print("Job:", os.environ.get("SLURM_JOB_ID", "local"))
print("Node:", os.uname().nodename)
print("Conditions:", CONDITIONS)
print("Output:", OUTPUT)


# ==============================================================
# Load conditions
# ==============================================================

conditions = []

with open(CONDITIONS, "r") as f:
    for line in f:
        line = line.strip()
        if line:
            conditions.append(json.loads(line))

print("\nTotal condition records:", len(conditions))

expected_conditions = {
    "C1_real",
    "C2_mismatched",
    "C3_blank",
    "C4_metadata_only",
    "C5_no_input",
}

actual_conditions = {r["condition"] for r in conditions}

N_EXPECTED = int(os.environ.get("PROJECT_A_N", 20))
if len(conditions) != 5 * N_EXPECTED:
    raise RuntimeError(
        f"Expected 100 pilot records, got {len(conditions)}"
    )

if actual_conditions != expected_conditions:
    raise RuntimeError(
        f"Unexpected conditions: {actual_conditions}"
    )

target_studies = {
    str(r["target_study_id"])
    for r in conditions
}

if len(target_studies) != N_EXPECTED:
    raise RuntimeError(
        f"Expected 20 target studies, got {len(target_studies)}"
    )

print("Target studies:", len(target_studies))


# ==============================================================
# Resume support
#
# A condition is considered complete only if a successful record
# with the same target_study_id + condition already exists.
# ==============================================================

completed = set()

if os.path.exists(OUTPUT):

    print("\nExisting output found. Checking completed conditions...")

    with open(OUTPUT, "r") as f:

        for line in f:

            try:
                record = json.loads(line)

                if record.get("status") != "success":
                    continue

                key = (
                    str(record["target_study_id"]),
                    record["condition"]
                )

                completed.add(key)

            except Exception:
                pass

print("Already completed:", len(completed))
print("Remaining:", len(conditions) - len(completed))


# ==============================================================
# Load model
# ==============================================================

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


# ==============================================================
# Helper: load image
# ==============================================================

def load_condition_image(row):

    condition = row["condition"]

    # ----------------------------------------------------------
    # C1 / C2 — real image file
    # ----------------------------------------------------------

    if condition in {"C1_real", "C2_mismatched"}:

        image_path = row["image_path"]

        if not image_path:
            raise RuntimeError(
                f"{condition}: image_path is empty"
            )

        # Paths in the condition file are relative to the MIMIC-CXR-JPG root
        dataset_root = os.environ.get(
            "MIMIC_CXR_ROOT",
            os.path.join(PROJECT_ROOT, "data", "mimic-cxr-jpg"),
        )

        full_path = os.path.join(
            dataset_root,
            image_path
        )

        if not os.path.isfile(full_path):
            raise FileNotFoundError(
                f"Image not found: {full_path}"
            )

        return Image.open(full_path).convert("RGB")

    # ----------------------------------------------------------
    # C3 — uniform black image
    # ----------------------------------------------------------

    if condition == "C3_blank":

        value = BLANK_VALUE

        return Image.new(
            "RGB",
            BLANK_SIZE,
            (value, value, value)
        )

    # ----------------------------------------------------------
    # C4 / C5 — no image
    # ----------------------------------------------------------

    if condition in {"C4_metadata_only", "C5_no_input"}:
        return None

    raise RuntimeError(
        f"Unknown condition: {condition}"
    )


# ==============================================================
# Helper: construct prompt
# ==============================================================

def build_prompt(row):

    condition = row["condition"]
    context_text = row.get("context_text", "") or ""

    # ----------------------------------------------------------
    # C4 — metadata only
    # ----------------------------------------------------------

    if condition == "C4_metadata_only":

        if not context_text:
            raise RuntimeError(
                f"C4 has no context_text for study "
                f"{row['target_study_id']}"
            )

        return (
            context_text
            + "\n\n"
            + PROMPT
        )

    # ----------------------------------------------------------
    # C5 — no image / no context
    # ----------------------------------------------------------

    if condition == "C5_no_input":

        return PROMPT

    # ----------------------------------------------------------
    # C1 / C2 / C3 — image conditions
    # ----------------------------------------------------------

    return PROMPT


# ==============================================================
# Inference
# ==============================================================

start_all = time.time()

with open(OUTPUT, "a") as fout:

    for idx, row in enumerate(conditions, start=1):

        target_study_id = str(row["target_study_id"])
        condition = row["condition"]

        key = (
            target_study_id,
            condition
        )

        if key in completed:
            continue

        print()
        print("=" * 70)
        print(
            f"[{idx}/{len(conditions)}] "
            f"Study {target_study_id} | "
            f"{condition}"
        )

        t0 = time.time()

        image = None
        image_path = row.get("image_path")

        try:

            # --------------------------------------------------
            # Prepare image
            # --------------------------------------------------

            image = load_condition_image(row)

            # --------------------------------------------------
            # Prepare prompt
            # --------------------------------------------------

            condition_prompt = build_prompt(row)

            # --------------------------------------------------
            # Build messages
            #
            # Image token is included ONLY for C1/C2/C3.
            # C4/C5 are genuinely text-only.
            # --------------------------------------------------

            if image is not None:

                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": condition_prompt
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
                )

            else:

                messages = [{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": condition_prompt
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
                    return_tensors="pt"
                )

            inputs = inputs.to(model.device)

            # --------------------------------------------------
            # Generate
            # --------------------------------------------------

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

            # --------------------------------------------------
            # Record
            # --------------------------------------------------

            record = {
                "target_study_id": int(row["target_study_id"]),
                "target_subject_id": int(row["target_subject_id"]),
                "source_study_id": (
                    int(row["source_study_id"])
                    if row.get("source_study_id") is not None
                    else None
                ),
                "source_subject_id": (
                    int(row["source_subject_id"])
                    if row.get("source_subject_id") is not None
                    else None
                ),
                "condition": condition,
                "image_type": row["image_type"],
                "image_path": image_path,
                "context": row.get("context", {}),
                "context_text": row.get("context_text", ""),
                "prompt": condition_prompt,
                "prediction": response,
                "inference_time_sec": round(elapsed, 2),
                "status": "success"
            }

            fout.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            fout.flush()

            completed.add(key)

            print("Prediction:", response)
            print("Time:", round(elapsed, 2), "sec")

        except Exception as e:

            elapsed = time.time() - t0

            print("ERROR:", repr(e))

            record = {
                "target_study_id": int(row["target_study_id"]),
                "target_subject_id": int(row["target_subject_id"]),
                "source_study_id": (
                    int(row["source_study_id"])
                    if row.get("source_study_id") is not None
                    else None
                ),
                "source_subject_id": (
                    int(row["source_subject_id"])
                    if row.get("source_subject_id") is not None
                    else None
                ),
                "condition": condition,
                "image_type": row["image_type"],
                "image_path": image_path,
                "context": row.get("context", {}),
                "context_text": row.get("context_text", ""),
                "prompt": (
                    build_prompt(row)
                    if condition in expected_conditions
                    else PROMPT
                ),
                "prediction": None,
                "inference_time_sec": round(elapsed, 2),
                "status": "error",
                "error": repr(e)
            }

            fout.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

            fout.flush()


# ==============================================================
# Final summary
# ==============================================================

elapsed_all = time.time() - start_all

print("\n=== PROJECT A PILOT COMPLETE ===")
print(
    "Total runtime:",
    round(elapsed_all / 3600, 2),
    "hours"
)
print("Output:", OUTPUT)

print("\nSuccessful conditions:")

successful = {}

if os.path.exists(OUTPUT):

    with open(OUTPUT, "r") as f:

        for line in f:

            try:
                r = json.loads(line)

                if r.get("status") == "success":

                    c = r["condition"]
                    successful[c] = successful.get(c, 0) + 1

            except Exception:
                pass

for condition in [
    "C1_real",
    "C2_mismatched",
    "C3_blank",
    "C4_metadata_only",
    "C5_no_input",
]:

    print(
        f"  {condition}: "
        f"{successful.get(condition, 0)}/20"
    )
