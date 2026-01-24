#!/usr/bin/env python3
"""
Lightweight schema validation for logs and results.
Prints PASS/FAIL and exits non-zero on failure.
"""

import json
import os
import sys
from typing import List

from config import LOG_DIR, RESULTS_FILE


def _collect_log_files() -> List[str]:
    if not os.path.isdir(LOG_DIR):
        return []
    return [
        os.path.join(LOG_DIR, name)
        for name in os.listdir(LOG_DIR)
        if name.endswith(".json") and "_session_" in name
    ]


def _validate_logs() -> List[str]:
    errors = []
    log_files = _collect_log_files()
    if not log_files:
        errors.append(f"Log check: no log files found in {LOG_DIR}.")
        return errors

    required_log_keys = {"model_name", "true_genre", "conversation"}
    for path in log_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            errors.append(f"Log check: failed to parse {path}: {exc}")
            continue

        missing = required_log_keys - set(data.keys())
        if missing:
            errors.append(f"Log check: {path} missing keys {sorted(missing)}")
            continue

        convo = data.get("conversation")
        if not isinstance(convo, list):
            errors.append(f"Log check: {path} conversation is not a list")
            continue

        for idx, turn in enumerate(convo):
            if not isinstance(turn, dict):
                errors.append(f"Log check: {path} turn {idx} is not a dict")
                continue
            speaker = turn.get("speaker")
            if speaker == "USER":
                if "text" not in turn:
                    errors.append(f"Log check: {path} USER turn {idx} missing text")
                user_meta = turn.get("user_meta")
                if user_meta is not None and not isinstance(user_meta, dict):
                    errors.append(f"Log check: {path} USER turn {idx} user_meta not a dict")
            elif speaker == "SYSTEM":
                if "text" not in turn:
                    errors.append(f"Log check: {path} SYSTEM turn {idx} missing text")
                if "constraints" not in turn:
                    errors.append(f"Log check: {path} SYSTEM turn {idx} missing constraints")
            else:
                errors.append(f"Log check: {path} turn {idx} has unknown speaker '{speaker}'")

    return errors


def _validate_results() -> List[str]:
    errors = []
    if not os.path.isfile(RESULTS_FILE):
        errors.append(f"Results check: {RESULTS_FILE} not found.")
        return errors

    required_metrics = [
        "concept_overlap_mean",
        "weighted_constraint_similarity_mean",
        "copying_penalty_mean",
        "trajectory_adaptation_score_mean",
        "concept_recovery_rate",
        "avg_recovery_delay",
        "missing_concepts_total",
        "hallucinated_concepts_total",
    ]

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                errors.append(f"Results check: line {line_num} JSON parse error: {exc}")
                continue

            if "model" not in record:
                errors.append(f"Results check: line {line_num} missing 'model'")
            judge = record.get("judge")
            if not isinstance(judge, dict):
                errors.append(f"Results check: line {line_num} missing or invalid 'judge' dict")

            for key in required_metrics:
                if key not in record:
                    errors.append(f"Results check: line {line_num} missing metric '{key}'")

    return errors


def main() -> int:
    errors = []
    errors.extend(_validate_logs())
    errors.extend(_validate_results())
    if errors:
        print("FAIL")
        for err in errors:
            print(f"- {err}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
