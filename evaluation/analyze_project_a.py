import csv
import json
import math
import os
import re
from collections import Counter, defaultdict
from itertools import combinations

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT = os.path.join(ROOT, "results", "project_a_lingshu_pilot.jsonl")
CLEAN_OUTPUT = os.path.join(ROOT, "results", "project_a_lingshu_pilot_clean.jsonl")
PAIRWISE_OUTPUT = os.path.join(ROOT, "results", "project_a_pairwise_similarity.csv")
SUMMARY_OUTPUT = os.path.join(ROOT, "results", "project_a_analysis_summary.json")

EXPECTED_CONDITIONS = [
    "C1_real",
    "C2_mismatched",
    "C3_blank",
    "C4_metadata_only",
    "C5_no_input",
]

PAIRWISE_PAIRS = [
    ("C1_real", "C2_mismatched"),
    ("C1_real", "C3_blank"),
    ("C1_real", "C4_metadata_only"),
    ("C1_real", "C5_no_input"),
    ("C2_mismatched", "C3_blank"),
    ("C3_blank", "C5_no_input"),
    ("C4_metadata_only", "C5_no_input"),
]


# ------------------------------------------------------------
# Text utilities
# ------------------------------------------------------------

def normalize_text(text):
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def token_jaccard(a, b):
    sa = set(tokenize(a))
    sb = set(tokenize(b))

    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0

    return len(sa & sb) / len(sa | sb)


def cosine_tf(a, b):
    ta = Counter(tokenize(a))
    tb = Counter(tokenize(b))

    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0

    vocab = set(ta) | set(tb)

    dot = sum(ta[x] * tb[x] for x in vocab)
    na = math.sqrt(sum(v * v for v in ta.values()))
    nb = math.sqrt(sum(v * v for v in tb.values()))

    if na == 0 or nb == 0:
        return 0.0

    return dot / (na * nb)


def rouge_l_f1(a, b):
    """
    Token-level ROUGE-L F1 using LCS.
    """
    x = tokenize(a)
    y = tokenize(b)

    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0

    # Dynamic-programming LCS.
    prev = [0] * (len(y) + 1)

    for xi in x:
        cur = [0] * (len(y) + 1)

        for j, yj in enumerate(y, start=1):
            if xi == yj:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = max(prev[j], cur[j - 1])

        prev = cur

    lcs = prev[-1]

    precision = lcs / len(x)
    recall = lcs / len(y)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def exact_match(a, b):
    return normalize_text(a) == normalize_text(b)


# ------------------------------------------------------------
# Input loading / deduplication
# ------------------------------------------------------------

def load_raw():
    records = []

    with open(INPUT, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON at line {line_no}: {e}"
                )

            records.append(record)

    return records


def clean_successes(records):
    """
    Raw output contains old error records plus the later successful
    retries. Keep exactly one SUCCESS record per
    (target_study_id, condition).
    """

    successes = defaultdict(list)

    for r in records:
        if r.get("status") != "success":
            continue

        key = (
            str(r.get("target_study_id")),
            r.get("condition"),
        )

        successes[key].append(r)

    duplicate_success_keys = {
        key: rows for key, rows in successes.items()
        if len(rows) > 1
    }

    if duplicate_success_keys:
        raise RuntimeError(
            "Found multiple successful records for the same "
            "(target_study_id, condition): "
            + str(list(duplicate_success_keys.keys()))
        )

    clean = [
        rows[0]
        for rows in successes.values()
    ]

    clean.sort(
        key=lambda r: (
            str(r.get("target_study_id")),
            EXPECTED_CONDITIONS.index(r.get("condition"))
            if r.get("condition") in EXPECTED_CONDITIONS
            else 999,
        )
    )

    return clean


# ------------------------------------------------------------
# Structural validation
# ------------------------------------------------------------

def validate_clean(records):
    errors = []

    keys = set()

    for r in records:
        target = str(r.get("target_study_id"))
        condition = r.get("condition")
        key = (target, condition)

        if key in keys:
            errors.append(f"Duplicate key: {key}")

        keys.add(key)

        if r.get("status") != "success":
            errors.append(f"Non-success record in clean data: {key}")

        prediction = r.get("prediction")

        if prediction is None:
            errors.append(f"Missing prediction: {key}")

    targets = sorted({
        str(r.get("target_study_id"))
        for r in records
    })

    if len(targets) != 20:
        errors.append(
            f"Expected 20 target studies, found {len(targets)}"
        )

    condition_counts = Counter(
        r.get("condition")
        for r in records
    )

    for condition in EXPECTED_CONDITIONS:
        if condition_counts[condition] != 20:
            errors.append(
                f"{condition}: expected 20, "
                f"found {condition_counts[condition]}"
            )

    for target in targets:
        observed = {
            r.get("condition")
            for r in records
            if str(r.get("target_study_id")) == target
        }

        missing = set(EXPECTED_CONDITIONS) - observed

        if missing:
            errors.append(
                f"{target}: missing conditions {sorted(missing)}"
            )

    if errors:
        raise RuntimeError(
            "\n".join(["Clean-data validation failed:"] + errors)
        )

    return {
        "records": len(records),
        "target_studies": len(targets),
        "condition_counts": dict(condition_counts),
    }


# ------------------------------------------------------------
# Basic response statistics
# ------------------------------------------------------------

def response_statistics(records):
    by_condition = defaultdict(list)

    for r in records:
        condition = r["condition"]
        prediction = r.get("prediction", "")

        by_condition[condition].append(prediction)

    result = {}

    for condition in EXPECTED_CONDITIONS:
        responses = by_condition[condition]

        token_lengths = [
            len(tokenize(x))
            for x in responses
        ]

        char_lengths = [
            len(normalize_text(x))
            for x in responses
        ]

        exact_duplicates = (
            len(responses)
            - len(set(normalize_text(x) for x in responses))
        )

        result[condition] = {
            "n": len(responses),
            "mean_tokens": (
                sum(token_lengths) / len(token_lengths)
                if token_lengths else 0
            ),
            "median_tokens": (
                sorted(token_lengths)[len(token_lengths) // 2]
                if token_lengths else 0
            ),
            "min_tokens": min(token_lengths) if token_lengths else 0,
            "max_tokens": max(token_lengths) if token_lengths else 0,
            "mean_characters": (
                sum(char_lengths) / len(char_lengths)
                if char_lengths else 0
            ),
            "exact_duplicate_count": exact_duplicates,
            "empty_count": sum(
                1 for x in responses
                if not normalize_text(x)
            ),
        }

    return result


# ------------------------------------------------------------
# Pairwise condition analysis
# ------------------------------------------------------------

def build_lookup(records):
    lookup = {}

    for r in records:
        key = (
            str(r["target_study_id"]),
            r["condition"],
        )
        lookup[key] = r

    return lookup


def pairwise_analysis(records):
    lookup = build_lookup(records)

    rows = []

    for condition_a, condition_b in PAIRWISE_PAIRS:

        for target in sorted({
            str(r["target_study_id"])
            for r in records
        }):

            a = lookup[(target, condition_a)]
            b = lookup[(target, condition_b)]

            text_a = a.get("prediction", "")
            text_b = b.get("prediction", "")

            rows.append({
                "target_study_id": target,
                "condition_a": condition_a,
                "condition_b": condition_b,
                "exact_match": int(exact_match(text_a, text_b)),
                "token_jaccard": token_jaccard(text_a, text_b),
                "tf_cosine": cosine_tf(text_a, text_b),
                "rouge_l_f1": rouge_l_f1(text_a, text_b),
                "tokens_a": len(tokenize(text_a)),
                "tokens_b": len(tokenize(text_b)),
            })

    return rows


def aggregate_pairwise(rows):
    grouped = defaultdict(list)

    for row in rows:
        key = (
            row["condition_a"],
            row["condition_b"],
        )
        grouped[key].append(row)

    result = {}

    for key, values in grouped.items():

        def mean(field):
            vals = [float(x[field]) for x in values]
            return sum(vals) / len(vals)

        exact_count = sum(
            x["exact_match"]
            for x in values
        )

        result[f"{key[0]}__vs__{key[1]}"] = {
            "n": len(values),
            "exact_match_rate": exact_count / len(values),
            "mean_token_jaccard": mean("token_jaccard"),
            "mean_tf_cosine": mean("tf_cosine"),
            "mean_rouge_l_f1": mean("rouge_l_f1"),
        }

    return result


# ------------------------------------------------------------
# Save outputs
# ------------------------------------------------------------

def save_clean(records):
    with open(CLEAN_OUTPUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False
                ) + "\n"
            )


def save_pairwise(rows):
    if not rows:
        return

    fields = list(rows[0].keys())

    with open(
        PAIRWISE_OUTPUT,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=== PROJECT A ANALYSIS ===")
    print()

    if not os.path.exists(INPUT):
        raise FileNotFoundError(
            f"Missing input: {INPUT}"
        )

    raw = load_raw()

    print("[1] Raw results")
    print(f"  Physical records: {len(raw)}")

    raw_status = Counter(
        r.get("status")
        for r in raw
    )

    print(f"  Status: {dict(raw_status)}")

    clean = clean_successes(raw)

    print()
    print("[2] Clean successful results")
    print(f"  Unique experiment cases: {len(clean)}")

    validation = validate_clean(clean)

    print(f"  Target studies: {validation['target_studies']}")
    print(
        f"  Conditions: "
        f"{validation['condition_counts']}"
    )

    save_clean(clean)

    print(f"  Saved: {CLEAN_OUTPUT}")

    stats = response_statistics(clean)

    print()
    print("[3] Response statistics")

    for condition in EXPECTED_CONDITIONS:
        s = stats[condition]

        print(
            f"  {condition}: "
            f"mean_tokens={s['mean_tokens']:.1f}, "
            f"median_tokens={s['median_tokens']}, "
            f"duplicates={s['exact_duplicate_count']}, "
            f"empty={s['empty_count']}"
        )

    pairwise_rows = pairwise_analysis(clean)
    pairwise_summary = aggregate_pairwise(pairwise_rows)

    save_pairwise(pairwise_rows)

    print()
    print("[4] Pairwise output similarity")

    for pair, s in pairwise_summary.items():

        print(
            f"  {pair}: "
            f"Jaccard={s['mean_token_jaccard']:.3f}, "
            f"TF-cosine={s['mean_tf_cosine']:.3f}, "
            f"ROUGE-L={s['mean_rouge_l_f1']:.3f}, "
            f"exact={s['exact_match_rate']:.3f}"
        )

    summary = {
        "input": INPUT,
        "clean_output": CLEAN_OUTPUT,
        "raw_records": len(raw),
        "clean_records": len(clean),
        "validation": validation,
        "response_statistics": stats,
        "pairwise_similarity": pairwise_summary,
    }

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("[5] Saved analysis summary")
    print(f"  {SUMMARY_OUTPUT}")
    print(f"  {PAIRWISE_OUTPUT}")

    print()
    print("=" * 60)
    print("ANALYSIS SETUP COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
