#!/usr/bin/env python3
"""Generate single-conversation 3D bar plots per model for TAS and components.

Outputs are written to figures/<mode>/3D_bars_run_0126 without touching
existing logs or figures.
"""

import json
import os
import random
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from config import RESULTS_FILE, FIG_DIR, CONCEPT_FIELDS


RNG_SEED = 126
OUT_SUBDIR = "3D_bars_run_0126"


def _load_results(path: str) -> List[Dict[str, object]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _pick_one_per_model(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    by_model: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        model = row.get("model")
        if not model:
            continue
        by_model.setdefault(str(model), []).append(row)
    rng = random.Random(RNG_SEED)
    picked = {}
    for model, entries in by_model.items():
        if entries:
            picked[model] = rng.choice(entries)
    return picked


def _field_weights(field_idx: int, n_fields: int, sigma: float = 0.9) -> np.ndarray:
    dist = np.abs(np.arange(n_fields, dtype=float) - float(field_idx))
    weights = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    if np.max(weights) > 0:
        weights = weights / np.max(weights)
    return weights


def _normalize_unit(z: np.ndarray) -> np.ndarray:
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return z
    z_min = float(np.min(finite))
    z_max = float(np.max(finite))
    if z_max <= z_min:
        return np.zeros_like(z)
    return (z - z_min) / (z_max - z_min)


def _plot_bars(model: str, turns: int, fields: List[str], z: np.ndarray, label: str, out_path: str):
    if z.size == 0:
        return
    z_scaled = _normalize_unit(z)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection="3d")
    base_cmap = plt.get_cmap("tab10" if len(fields) <= 10 else "tab20")
    colors = [base_cmap(i % base_cmap.N) for i in range(len(fields))]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(fields) + 0.5, 1.0), cmap.N)

    # Build bar positions
    n_fields = len(fields)
    peak_fields = np.argmax(z_scaled, axis=0)
    dx_grey = 0.7
    dy_grey = 0.6
    dx_peak = 1.0
    dy_peak = 0.7

    xs_p, ys_p, zs_p, colors_p = [], [], [], []
    for t_idx in range(turns):
        for f_idx in range(n_fields):
            z_val = z_scaled[f_idx, t_idx]
            if f_idx == peak_fields[t_idx]:
                xs_p.append(float(t_idx + 1) - dx_peak / 2)
                ys_p.append(float(f_idx) - dy_peak / 2)
                zs_p.append(z_val)
                colors_p.append(cmap(norm(f_idx)))
            else:
                continue

    if xs_p:
        xs_p = np.array(xs_p)
        ys_p = np.array(ys_p)
        zs_p = np.array(zs_p)
        ax.bar3d(xs_p, ys_p, np.zeros_like(zs_p), dx_peak, dy_peak, zs_p, color=colors_p, shade=True, linewidth=0.0)

    ax.set_xlim(1.0, float(turns))
    ax.set_ylim(-0.5, float(n_fields - 0.5))
    ax.set_zlim(0.0, 1.0)
    ax.set_zticks([0.0, 1.0])
    try:
        ax.set_box_aspect((1.0, 1.0, 0.35))
    except Exception:
        pass

    ax.set_xlabel("Turn", labelpad=24)
    ax.set_ylabel("Focus field", labelpad=24)
    ax.set_zlabel("Score (scaled 0-1)", labelpad=24)
    ax.set_title(f"{label} bars (single conversation) - {model}", pad=20)

    tick_count = min(6, max(3, turns // 5))
    xticks = np.linspace(1, max(2, turns), tick_count, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])

    field_idx = np.arange(n_fields)
    ax.set_yticks(field_idx)
    ax.set_yticklabels(fields, fontsize=12)
    for tick_label, idx in zip(ax.get_yticklabels(), field_idx):
        tick_label.set_color(cmap(norm(idx)))

    ax.tick_params(axis="x", pad=12)
    ax.tick_params(axis="y", pad=14)
    ax.tick_params(axis="z", pad=12)

    # Legend removed per request; focus field colors shown on tick labels.

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    rows = _load_results(RESULTS_FILE)
    if not rows:
        raise SystemExit(f"No results found at {RESULTS_FILE}. Run batch_run.py first.")

    picked = _pick_one_per_model(rows)
    if not picked:
        raise SystemExit("No model entries found in results.")

    out_dir = os.path.join(FIG_DIR, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    fields = list(CONCEPT_FIELDS) + ["other"]
    metrics = {
        "TAS": "tas",
        "CO": "cc",
        "WCS": "cr",
        "CP": "i",
    }

    for model, entry in picked.items():
        detail = entry.get("detail") or {}
        tl = detail.get("turn_level") or []
        if not isinstance(tl, list) or not tl:
            continue
        turns = len(tl)
        field_to_idx = {f: i for i, f in enumerate(fields)}
        for label, key in metrics.items():
            z = np.zeros((len(fields), turns), dtype=float)
            for t_idx, row in enumerate(tl):
                if not isinstance(row, dict):
                    continue
                meta = row.get("user_meta") or {}
                focus = meta.get("focus_field") if isinstance(meta, dict) else None
                focus_field = focus if focus in field_to_idx else "other"
                try:
                    score = float(row.get(key))
                except (TypeError, ValueError):
                    score = 0.0
                weights = _field_weights(field_to_idx[focus_field], len(fields))
                z[:, t_idx] = score * weights

            out_path = os.path.join(
                out_dir, f"3d_bars_{label.lower()}_{model.replace(':', '_')}.png"
            )
            _plot_bars(model, turns, fields, z, label, out_path)


if __name__ == "__main__":
    main()
