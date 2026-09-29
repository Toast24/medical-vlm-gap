#!/usr/bin/env python3

import json
import os
from collections import Counter, defaultdict

CONDITIONS = "data/project_a_pilot_conditions.jsonl"
RESULTS = "results/project_a_lingshu_pilot_clean.jsonl"
REFERENCES = "data/mimic_cxr_test_references_v2.jsonl"
IMAGE_ROOT = "data/mimic-cxr-jpg"

EXPECTED = {
    "C1_real",
    "C2_mismatched",
    "C3_blank",
    "C4_metadata_only",
    "C5_no_input",
}

PROMPT = (
    "Describe this chest X-ray briefly. "
    "Focus on the main radiographic findings and abnormalities. "
    "Do not speculate beyond what is visible."
)

errors = []
warnings = []


def fail(msg):
    errors.append(msg)


def warn(msg):
    warnings.append(msg)


# ============================================================
# Load data
# ============================================================

def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


conditions = load_jsonl(CONDITIONS)
results = load_jsonl(RESULTS)
references = load_jsonl(REFERENCES)

print("=== PROJECT A VALIDATION ===\n")


# ============================================================
# 1. Condition artifact
# ============================================================

print("[1] Condition artifact")

if len(conditions) != 100:
    fail(f"Expected 100 condition rows, found {len(conditions)}")

condition_counts = Counter(r.get("condition") for r in conditions)

print("  Rows:", len(conditions))
print("  Conditions:", dict(condition_counts))

if set(condition_counts) != EXPECTED:
    fail(
        f"Unexpected condition set: {set(condition_counts)}"
    )

for c in EXPECTED:
    if condition_counts[c] != 20:
        fail(
            f"{c}: expected 20 rows, found {condition_counts[c]}"
        )


# ============================================================
# 2. Target study structure
# ============================================================

print("\n[2] Target study structure")

by_target = defaultdict(list)

for r in conditions:
    by_target[str(r["target_study_id"])].append(r)

print("  Target studies:", len(by_target))

if len(by_target) != 20:
    fail(
        f"Expected 20 target studies, found {len(by_target)}"
    )

for target, rows in sorted(by_target.items()):
    cs = {r["condition"] for r in rows}

    if cs != EXPECTED:
        fail(
            f"Study {target}: missing/unexpected conditions: "
            f"{EXPECTED - cs} / {cs - EXPECTED}"
        )

    if len(rows) != 5:
        fail(
            f"Study {target}: expected 5 rows, found {len(rows)}"
        )


# ============================================================
# 3. C1 image validation
# ============================================================

print("\n[3] C1 real-image validation")

c1 = [r for r in conditions if r["condition"] == "C1_real"]

for r in c1:
    if not r.get("image_path"):
        fail(
            f"C1 study {r['target_study_id']}: missing image_path"
        )
        continue

    path = os.path.join(IMAGE_ROOT, r["image_path"])

    if not os.path.isfile(path):
        fail(
            f"C1 study {r['target_study_id']}: image missing: {path}"
        )

print("  Checked:", len(c1))


# ============================================================
# 4. C2 mismatch validation
# ============================================================

print("\n[4] C2 mismatched-image validation")

c2 = [r for r in conditions if r["condition"] == "C2_mismatched"]

for r in c2:
    target = str(r["target_study_id"])
    source = str(r.get("source_study_id"))

    if not r.get("image_path"):
        fail(
            f"C2 study {target}: missing image_path"
        )

    else:
        path = os.path.join(IMAGE_ROOT, r["image_path"])

        if not os.path.isfile(path):
            fail(
                f"C2 study {target}: image missing: {path}"
            )

    if source == target:
        fail(
            f"C2 study {target}: source_study_id equals target_study_id"
        )

    target_subject = str(r.get("target_subject_id"))
    source_subject = str(r.get("source_subject_id"))

    if target_subject == source_subject:
        fail(
            f"C2 study {target}: source and target subject are identical"
        )

print("  Checked:", len(c2))


# ============================================================
# 5. C3 blank-image validation
# ============================================================

print("\n[5] C3 blank-image validation")

c3 = [r for r in conditions if r["condition"] == "C3_blank"]

for r in c3:
    if r.get("image_path"):
        fail(
            f"C3 study {r['target_study_id']}: "
            "should not have an image_path"
        )

print("  Checked:", len(c3))
print("  Expected generated image: 512x512 RGB, value 0")


# ============================================================
# 6. C4 metadata-only validation
# ============================================================

print("\n[6] C4 metadata-only validation")

c4 = [r for r in conditions if r["condition"] == "C4_metadata_only"]

for r in c4:
    target = r["target_study_id"]
    context_text = r.get("context_text", "") or ""

    if not context_text:
        fail(
            f"C4 study {target}: empty context_text"
        )

    if r.get("image_path"):
        fail(
            f"C4 study {target}: should not have image_path"
        )

    forbidden = [
        "finding",
        "impression",
        "comparison",
        "prior_report",
        "reference_text",
        "reason for exam",
        "indication",
    ]

    context = r.get("context", {})

    serialized = (
        json.dumps(context, ensure_ascii=False)
        + " "
        + context_text
    ).lower()

    for term in forbidden:
        if term in serialized:
            fail(
                f"C4 study {target}: forbidden term found: {term}"
            )

print("  Checked:", len(c4))


# ============================================================
# 7. C5 no-input validation
# ============================================================

print("\n[7] C5 no-input validation")

c5 = [r for r in conditions if r["condition"] == "C5_no_input"]

for r in c5:
    target = r["target_study_id"]

    if r.get("image_path"):
        fail(
            f"C5 study {target}: should not have image_path"
        )

    if r.get("context_text", ""):
        fail(
            f"C5 study {target}: should not have context_text"
        )

print("  Checked:", len(c5))


# ============================================================
# 8. Result artifact
# ============================================================

print("\n[8] Lingshu result validation")

if len(results) != 100:
    fail(
        f"Expected 100 clean results, found {len(results)}"
    )

result_keys = set()

for r in results:

    key = (
        str(r.get("target_study_id")),
        r.get("condition")
    )

    if key in result_keys:
        fail(f"Duplicate result: {key}")

    result_keys.add(key)

    if r.get("status") != "success":
        fail(
            f"Non-success result: {key}"
        )

expected_keys = {
    (
        str(r["target_study_id"]),
        r["condition"]
    )
    for r in conditions
}

missing_results = expected_keys - result_keys
extra_results = result_keys - expected_keys

if missing_results:
    fail(
        f"Missing results: {sorted(missing_results)}"
    )

if extra_results:
    fail(
        f"Unexpected results: {sorted(extra_results)}"
    )

result_counts = Counter(r.get("condition") for r in results)

print("  Results:", len(results))
print("  By condition:", dict(result_counts))


# ============================================================
# 9. Prompt consistency
# ============================================================

print("\n[9] Prompt consistency")

for r in results:

    condition = r["condition"]
    prompt = r.get("prompt", "")

    if condition == "C4_metadata_only":
        context_text = r.get("context_text", "") or ""

        expected_prompt = context_text + "\n\n" + PROMPT

        if prompt != expected_prompt:
            fail(
                f"C4 study {r['target_study_id']}: "
                "stored prompt does not match construction"
            )

    elif condition == "C5_no_input":
        if prompt != PROMPT:
            fail(
                f"C5 study {r['target_study_id']}: "
                "prompt differs from base prompt"
            )

    else:
        if prompt != PROMPT:
            fail(
                f"{condition} study {r['target_study_id']}: "
                "prompt differs from base prompt"
            )


# ============================================================
# 10. Reference artifact
# ============================================================

print("\n[10] Reference artifact")

reference_by_study = {}

for r in references:

    study = str(
        r.get("study_id", r.get("target_study_id", ""))
    )

    if not study:
        warn("Reference row without study_id/target_study_id")
        continue

    if study in reference_by_study:
        warn(f"Multiple reference rows for study {study}")

    reference_by_study[study] = r

target_ids = set(by_target)

missing_refs = sorted(
    target_ids - set(reference_by_study)
)

print("  Reference rows:", len(references))
print("  Target references found:", len(target_ids) - len(missing_refs))

if missing_refs:
    fail(
        f"Missing references for studies: {missing_refs}"
    )


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION SUMMARY")
print("=" * 60)

print("Errors:", len(errors))
print("Warnings:", len(warnings))

if errors:
    print("\nFAILURES:")
    for e in errors:
        print("  -", e)

if warnings:
    print("\nWARNINGS:")
    for w in warnings:
        print("  -", w)

if not errors:
    print("\nOVERALL: PASS")
    print("Project A pilot is structurally valid for evaluation.")
else:
    print("\nOVERALL: FAIL")
    print("Fix the listed issues before computing metrics.")

raise SystemExit(1 if errors else 0)
