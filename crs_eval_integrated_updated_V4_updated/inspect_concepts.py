#!/usr/bin/env python3
"""inspect_concepts.py

Utility to inspect extracted concepts to confirm they are not dominated by useless/common tokens.

Reads results.jsonl and aggregates:
- Top system concepts overall and by model (if available)
- Top missing/hallucinated concepts (from turn-level or segment-level detail)
- Top plot keywords (`plot_kw=...`) to spot contamination

Outputs CSVs under figures/concept_inspection/.
"""

import json
import os
from collections import Counter, defaultdict

import pandas as pd

from config import RESULTS_FILE, FIG_DIR, USE_PLOT_KW_FOR_TAS

OUT_DIR = os.path.join(FIG_DIR, "concept_inspection")

def main():
    if not os.path.exists(RESULTS_FILE):
        raise SystemExit(f"Missing {RESULTS_FILE}. Run evaluations first.")
    os.makedirs(OUT_DIR, exist_ok=True)

    system_overall = Counter()
    system_by_model = defaultdict(Counter)
    plot_overall = Counter()
    missing_overall = Counter()
    hallucinated_overall = Counter()
    missing_by_model = defaultdict(Counter)
    hallucinated_by_model = defaultdict(Counter)

    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            model = rec.get("model_name") or rec.get("model") or "unknown"
            detail = rec.get("detail") or {}
            turns = detail.get("turn_concepts") or []
            for t in turns:
                for c in t.get("system_concepts", []):
                    system_overall[c] += 1
                    system_by_model[model][c] += 1
                    if c.startswith("plot=") or c.startswith("plot_kw="):
                        plot_overall[c] += 1

            turn_level = detail.get("turn_level") or []
            if isinstance(turn_level, list) and turn_level:
                for t in turn_level:
                    for c in t.get("missing_concepts", []):
                        missing_overall[c] += 1
                        missing_by_model[model][c] += 1
                        if c.startswith("plot=") or c.startswith("plot_kw="):
                            plot_overall[c] += 1
                    for c in t.get("hallucinated_concepts", []):
                        hallucinated_overall[c] += 1
                        hallucinated_by_model[model][c] += 1
                        if c.startswith("plot=") or c.startswith("plot_kw="):
                            plot_overall[c] += 1
            else:
                segments = detail.get("segments") or []
                for seg in segments:
                    for c in seg.get("missing_concepts", []):
                        missing_overall[c] += 1
                        missing_by_model[model][c] += 1
                    for c in seg.get("hallucinated_concepts", []):
                        hallucinated_overall[c] += 1
                        hallucinated_by_model[model][c] += 1

    if system_overall:
        pd.DataFrame(system_overall.most_common(200), columns=["concept", "count"]).to_csv(
            os.path.join(OUT_DIR, "top_system_concepts_overall.csv"), index=False
        )
    if plot_overall and USE_PLOT_KW_FOR_TAS:
        pd.DataFrame(plot_overall.most_common(200), columns=["plot_concept", "count"]).to_csv(
            os.path.join(OUT_DIR, "top_plot_concepts_overall.csv"), index=False
        )
    if missing_overall:
        pd.DataFrame(missing_overall.most_common(200), columns=["concept", "count"]).to_csv(
            os.path.join(OUT_DIR, "top_missing_concepts_overall.csv"), index=False
        )
    if hallucinated_overall:
        pd.DataFrame(hallucinated_overall.most_common(200), columns=["concept", "count"]).to_csv(
            os.path.join(OUT_DIR, "top_hallucinated_concepts_overall.csv"), index=False
        )

    rows = []
    for model, c in system_by_model.items():
        for concept, cnt in c.most_common(50):
            rows.append({"model": model, "concept": concept, "count": cnt})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "top_system_concepts_by_model.csv"), index=False)

    rows = []
    for model, c in missing_by_model.items():
        for concept, cnt in c.most_common(50):
            rows.append({"model": model, "concept": concept, "count": cnt})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "top_missing_concepts_by_model.csv"), index=False)

    rows = []
    for model, c in hallucinated_by_model.items():
        for concept, cnt in c.most_common(50):
            rows.append({"model": model, "concept": concept, "count": cnt})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "top_hallucinated_concepts_by_model.csv"), index=False)

    print(f"Wrote concept inspection CSVs to: {OUT_DIR}")

if __name__ == "__main__":
    main()
