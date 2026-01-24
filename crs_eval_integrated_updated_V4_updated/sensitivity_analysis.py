#!/usr/bin/env python3
"""sensitivity_analysis.py

Sensitivity analysis for TAS weights.

Reads results.jsonl and recomputes TAS under a grid of weights for the 3 components:
- CO (concept overlap)
- WCS (weighted constraint similarity)
- CP (copying penalty; lower is better)

Outputs:
- tables/sensitivity_grid.csv : mean TAS per model per weight setting
- figures/sensitivity_heatmap_cc_cr_<model>.png : heatmap over w_cc/w_cr (w_cp implied)
- figures/sensitivity_heatmap_cc_cp_<model>.png : heatmap over w_cc/w_cp (w_cr implied)
- figures/sensitivity_heatmap_cr_cp_<model>.png : heatmap over w_cr/w_cp (w_cc implied)

No external deps beyond numpy/pandas/matplotlib (already used in repo).
"""

import json
import os
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_FILE, FIG_DIR

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

def load_results(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(recs)
    if "detail" in df.columns:
        df["detail"] = df["detail"].apply(lambda x: x if isinstance(x, dict) else {})
    else:
        df["detail"] = [{} for _ in range(len(df))]
    return df

def extract_components(detail: dict) -> Tuple[List[float], List[float], List[float]]:
    ts = detail.get("turn_scores") or {}
    cc = ts.get("cc")
    cr = ts.get("cr")
    i = ts.get("i")
    if isinstance(cc, list) and isinstance(cr, list) and isinstance(i, list) and cc and cr and i:
        return [float(x) for x in cc], [float(x) for x in cr], [float(x) for x in i]

    tl = detail.get("turn_level") or []
    cc_list: List[float] = []
    cr_list: List[float] = []
    i_list: List[float] = []
    if isinstance(tl, list):
        for row in tl:
            if not isinstance(row, dict):
                continue
            try:
                cc_val = float(row.get("cc"))
                cr_val = float(row.get("cr"))
                i_val = float(row.get("i"))
            except (TypeError, ValueError):
                continue
            cc_list.append(cc_val)
            cr_list.append(cr_val)
            i_list.append(i_val)
    return cc_list, cr_list, i_list

def tas_from_components(cc: List[float], cr: List[float], i: List[float], w_cc: float, w_cr: float, w_cp: float) -> float:
    if not cc or not cr or not i:
        return 0.0
    cc_m = float(np.mean(cc))
    cr_m = float(np.mean(cr))
    cp_m = float(np.mean(i))
    return float((w_cc * cc_m) + (w_cr * cr_m) - (w_cp * cp_m))

def main():
    if not os.path.exists(RESULTS_FILE):
        raise SystemExit(f"Missing {RESULTS_FILE}. Run evaluations first.")

    os.makedirs(FIG_DIR, exist_ok=True)
    out_dir = os.path.join(FIG_DIR, "sensitivity")
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    df = load_results(RESULTS_FILE)
    model_col = "model_name" if "model_name" in df.columns else ("model" if "model" in df.columns else None)
    if model_col is None:
        raise SystemExit("Missing model column in results.")

    # grid over weights (step=0.1), gamma implied
    steps = [i / 10.0 for i in range(0, 11)]
    rows = []
    for a, b in itertools.product(steps, steps):
        if a + b > 1.0:
            continue
        c = 1.0 - a - b
        for _, r in df.iterrows():
            cc, cr, i = extract_components(r.get("detail", {}))
            tas = tas_from_components(cc, cr, i, a, b, c)
            rows.append({model_col: r[model_col], "w_cc": a, "w_cr": b, "w_cp": c, "tas": tas})

    grid = pd.DataFrame(rows)
    summary = grid.groupby([model_col, "w_cc", "w_cr", "w_cp"])["tas"].mean().reset_index()
    summary_csv = os.path.join(out_dir, "sensitivity_grid.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"Saved: {summary_csv}")

    def plot_heatmap(piv: pd.DataFrame, title: str, xlabel: str, ylabel: str, out_path: str, vmin: float, vmax: float):
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")
        data = np.ma.masked_invalid(piv.values.astype(float))
        plt.figure(figsize=(10, 8))
        plt.imshow(data, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.xticks(range(len(piv.columns)), [f"{x:.1f}" for x in piv.columns], rotation=90)
        plt.yticks(range(len(piv.index)), [f"{x:.1f}" for x in piv.index])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()

    # Heatmaps per model: w_cc vs w_cr, w_cc vs w_cp, w_cr vs w_cp
    for model in summary[model_col].unique():
        sub = summary[summary[model_col] == model]
        vmin = float(sub["tas"].min())
        vmax = float(sub["tas"].max())

        piv_cc_cr = sub.pivot(index="w_cc", columns="w_cr", values="tas")
        plot_heatmap(
            piv_cc_cr,
            f"TAS sensitivity: w_cc vs w_cr ({model})\n(w_cp implied = 1 - w_cc - w_cr)",
            "w_cr",
            "w_cc",
            os.path.join(out_dir, f"sensitivity_heatmap_cc_cr_{model}.png"),
            vmin,
            vmax,
        )

        piv_cc_cp = sub.pivot(index="w_cc", columns="w_cp", values="tas")
        plot_heatmap(
            piv_cc_cp,
            f"TAS sensitivity: w_cc vs w_cp ({model})\n(w_cr implied = 1 - w_cc - w_cp)",
            "w_cp",
            "w_cc",
            os.path.join(out_dir, f"sensitivity_heatmap_cc_cp_{model}.png"),
            vmin,
            vmax,
        )

        piv_cr_cp = sub.pivot(index="w_cr", columns="w_cp", values="tas")
        plot_heatmap(
            piv_cr_cp,
            f"TAS sensitivity: w_cr vs w_cp ({model})\n(w_cc implied = 1 - w_cr - w_cp)",
            "w_cp",
            "w_cr",
            os.path.join(out_dir, f"sensitivity_heatmap_cr_cp_{model}.png"),
            vmin,
            vmax,
        )

    print("--- Sensitivity analysis complete. ---")

if __name__ == "__main__":
    main()
