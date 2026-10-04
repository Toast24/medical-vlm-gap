import csv
import gzip
import json
import os
from pathlib import Path

PROJECT = Path(
    os.environ.get(
        "PROJECT_ROOT",
        Path(__file__).resolve().parents[1],
    )
)

import os as _os
N_EXPECTED = int(_os.environ.get("PROJECT_A_N", 20))
PILOT_PATH = PROJECT / _os.environ.get("PROJECT_A_PILOT", "data/project_a_pilot_20.jsonl")
MANIFEST_PATH = PROJECT / "data/mimic-cxr-jpg/mimic_cxr_test_manifest.csv"
METADATA_PATH = PROJECT / "data/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv.gz"
OUTPUT_PATH = PROJECT / _os.environ.get("PROJECT_A_CONDITIONS_OUT", "data/project_a_pilot_conditions.jsonl")

PROMPT = (
    "Describe this chest X-ray briefly. "
    "Focus on the main radiographic findings and abnormalities. "
    "Do not speculate beyond what is visible."
)

BLANK_IMAGE_SPEC = {
    "type": "uniform_black",
    "width": 512,
    "height": 512,
    "pixel_value": 0,
}

# Only genuine acquisition/procedure metadata.
C4_FIELDS = [
    "ProcedureCodeSequence_CodeMeaning",
    "ViewPosition",
    "ViewCodeSequence_CodeMeaning",
    "PatientOrientationCodeSequence_CodeMeaning",
]

# Values that are known placeholders or effectively missing.
INVALID_VALUES = {
    "",
    "performed desc",
    "performed description",
    "unknown",
    "n/a",
    "na",
    "none",
    "null",
}


def clean(value):
    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    if value.lower() in INVALID_VALUES:
        return None

    return value


def build_c4_context(meta):
    """
    Construct C4 from structured MIMIC-CXR acquisition metadata only.

    No report text, findings, impression, comparison, indication,
    or prior report is used.
    """

    context = {}

    for field in C4_FIELDS:
        value = clean(meta.get(field))

        if value is not None:
            context[field] = value

    return context


def context_to_text(context):
    """
    Deterministic textual serialization used by the inference harness.
    """

    labels = {
        "ProcedureCodeSequence_CodeMeaning": "Procedure",
        "ViewPosition": "View position",
        "ViewCodeSequence_CodeMeaning": "View",
        "PatientOrientationCodeSequence_CodeMeaning": "Patient orientation",
    }

    parts = []

    for field in C4_FIELDS:
        if field in context:
            parts.append(
                f"{labels[field]}: {context[field]}"
            )

    if not parts:
        return ""

    return (
        "The following structured imaging metadata is available "
        "before interpreting the image:\n"
        + "\n".join(parts)
    )


# ------------------------------------------------------------------
# Load pilot
# ------------------------------------------------------------------

with open(PILOT_PATH) as f:
    pilot_rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

if len(pilot_rows) != N_EXPECTED:
    raise RuntimeError(
        f"Expected 20 pilot studies, found {len(pilot_rows)}"
    )


# ------------------------------------------------------------------
# Load study/image manifest
# ------------------------------------------------------------------

manifest = {}

with open(MANIFEST_PATH, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        manifest[int(row["study_id"])] = row


# ------------------------------------------------------------------
# Load official MIMIC-CXR metadata
# ------------------------------------------------------------------

metadata = {}

with gzip.open(METADATA_PATH, "rt", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        metadata[row["dicom_id"]] = row


# ------------------------------------------------------------------
# Build C1-C5
# ------------------------------------------------------------------

conditions = []

for pilot in pilot_rows:

    target_study_id = int(pilot["target_study_id"])
    target_subject_id = int(pilot["target_subject_id"])

    source_study_id = int(pilot["source_study_id"])
    source_subject_id = int(pilot["source_subject_id"])

    if target_study_id not in manifest:
        raise RuntimeError(
            f"Missing target study {target_study_id}"
        )

    if source_study_id not in manifest:
        raise RuntimeError(
            f"Missing source study {source_study_id}"
        )

    target = manifest[target_study_id]
    source = manifest[source_study_id]

    target_dicom_id = target["primary_dicom_id"]
    source_dicom_id = source["primary_dicom_id"]

    if target_dicom_id not in metadata:
        raise RuntimeError(
            f"Missing metadata for target DICOM {target_dicom_id}"
        )

    if source_dicom_id not in metadata:
        raise RuntimeError(
            f"Missing metadata for source DICOM {source_dicom_id}"
        )

    target_meta = metadata[target_dicom_id]

    c4_context = build_c4_context(target_meta)
    c4_context_text = context_to_text(c4_context)

    base = {
        "target_study_id": target_study_id,
        "target_subject_id": target_subject_id,

        "source_study_id": source_study_id,
        "source_subject_id": source_subject_id,

        "target_image_path": target["primary_image_path"],
        "source_image_path": source["primary_image_path"],

        "reference_text": pilot["reference_text"],
        "reference_method": pilot["reference_method"],

        "prompt": PROMPT,

        "seed": 42,
    }

    # ==============================================================
    # C1 — Real image
    # ==============================================================

    conditions.append({
        **base,
        "condition": "C1_real",
        "image_type": "target",
        "image_path": target["primary_image_path"],
        "context": {},
        "context_text": "",
    })

    # ==============================================================
    # C2 — Mismatched image
    # ==============================================================

    if target_study_id == source_study_id:
        raise RuntimeError(
            f"C2 self-match detected: {target_study_id}"
        )

    conditions.append({
        **base,
        "condition": "C2_mismatched",
        "image_type": "mismatched",
        "image_path": source["primary_image_path"],
        "context": {},
        "context_text": "",
    })

    # ==============================================================
    # C3 — Blank image
    # ==============================================================

    conditions.append({
        **base,
        "condition": "C3_blank",
        "image_type": "blank",
        "image_path": None,
        "blank_image": BLANK_IMAGE_SPEC,
        "context": {},
        "context_text": "",
    })

    # ==============================================================
    # C4 — Structured metadata only
    # ==============================================================

    conditions.append({
        **base,
        "condition": "C4_metadata_only",
        "image_type": "none",
        "image_path": None,
        "context": c4_context,
        "context_text": c4_context_text,
    })

    # ==============================================================
    # C5 — No image / no context
    # ==============================================================

    conditions.append({
        **base,
        "condition": "C5_no_input",
        "image_type": "none",
        "image_path": None,
        "context": {},
        "context_text": "",
    })


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------

expected_counts = {
    "C1_real": N_EXPECTED,
    "C2_mismatched": N_EXPECTED,
    "C3_blank": N_EXPECTED,
    "C4_metadata_only": N_EXPECTED,
    "C5_no_input": N_EXPECTED,
}

if len(conditions) != 5 * N_EXPECTED:
    raise RuntimeError(
        f"Expected 100 records, got {len(conditions)}"
    )

counts = {}

for row in conditions:
    counts[row["condition"]] = (
        counts.get(row["condition"], 0) + 1
    )

if counts != expected_counts:
    raise RuntimeError(
        f"Unexpected condition counts: {counts}"
    )


# Every target has exactly five conditions.
by_target = {}

for row in conditions:
    by_target.setdefault(
        row["target_study_id"], []
    ).append(row)

for study_id, rows in by_target.items():

    if len(rows) != 5:
        raise RuntimeError(
            f"Study {study_id}: expected 5 conditions, "
            f"got {len(rows)}"
        )

    names = {
        row["condition"]
        for row in rows
    }

    if names != set(expected_counts):
        raise RuntimeError(
            f"Study {study_id}: incorrect conditions {names}"
        )


# C4 must not contain report-derived information.
for row in conditions:

    if row["condition"] != "C4_metadata_only":
        continue

    forbidden = [
        "findings",
        "impression",
        "comparison",
        "prior_report",
        "reference_text",
        "reason for exam",
        "indication",
    ]

    combined = (
        json.dumps(row["context"], ensure_ascii=False)
        + " "
        + row["context_text"]
    ).lower()

    for term in forbidden:
        if term in combined:
            raise RuntimeError(
                f"C4 leakage detected for study "
                f"{row['target_study_id']}: {term}"
            )


# C4 context must exactly match context_text serialization.
for row in conditions:

    if row["condition"] != "C4_metadata_only":
        continue

    expected_text = context_to_text(row["context"])

    if row["context_text"] != expected_text:
        raise RuntimeError(
            f"C4 context serialization mismatch for "
            f"study {row['target_study_id']}"
        )


# ------------------------------------------------------------------
# Write
# ------------------------------------------------------------------

with open(OUTPUT_PATH, "w") as f:
    for row in conditions:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            ) + "\n"
        )


# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------

c4 = [
    row
    for row in conditions
    if row["condition"] == "C4_metadata_only"
]

with_metadata = [
    row
    for row in c4
    if row["context"]
]

without_metadata = [
    row
    for row in c4
    if not row["context"]
]

print(f"Created: {OUTPUT_PATH}")
print(f"Studies: {len(pilot_rows)}")
print(f"Conditions: {len(conditions)}")

for condition in expected_counts:
    print(f"{condition}: {counts[condition]}")

print()
print(f"C4 with metadata: {len(with_metadata)}")
print(f"C4 without metadata: {len(without_metadata)}")

print()
print("C4 examples:")

for row in c4[:10]:
    print()
    print("Study:", row["target_study_id"])
    print("Context:", row["context"])
    print("Context text:")
    print(row["context_text"])

print()
print("Validation: PASSED")
