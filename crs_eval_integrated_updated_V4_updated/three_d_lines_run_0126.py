#!/usr/bin/env python3
"""Generate single-conversation 3D line plots per model for TAS and components.

Outputs are written to figures/<mode>/3D_lines_run_0126 without touching
existing logs or figures.
"""

import json
import os
import random
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from config import RESULTS_FILE, FIG_DIR, CONCEPT_FIELDS


RNG_SEED = 126
OUT_SUBDIR = "3D_lines_run_0126"


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


def _smooth_along_turns(z: np.ndarray) -> np.ndarray:
    kernel = np.array([0.2, 0.6, 0.2], dtype=float)
    out = np.zeros_like(z)
    for i in range(z.shape[0]):
        out[i, :] = np.convolve(z[i, :], kernel, mode="same")
    return out


def _upsample_axis(values: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return values
    n = values.shape[-1]
    if n < 2:
        return values
    x_old = np.linspace(0.0, 1.0, n)
    x_new = np.linspace(0.0, 1.0, n * factor)
    return np.interp(x_new, x_old, values)


def _normalize_unit(z: np.ndarray) -> np.ndarray:
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return z
    z_min = float(np.min(finite))
    z_max = float(np.max(finite))
    if z_max <= z_min:
        return np.zeros_like(z)
    return (z - z_min) / (z_max - z_min)


def _line_segments(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    pts = np.column_stack([x, y, z])
    return np.stack([pts[:-1], pts[1:]], axis=1)


def _plot_lines(model: str, turns: int, fields: List[str], z: np.ndarray, label: str, out_path: str):
    if z.size == 0:
        return
    z_scaled = _normalize_unit(z)
    z_scaled = _smooth_along_turns(z_scaled)
    z_scaled = _normalize_unit(z_scaled)

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    base_cmap = plt.get_cmap("tab10" if len(fields) <= 10 else "tab20")
    colors = [base_cmap(i % base_cmap.N) for i in range(len(fields))]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(fields) + 0.5, 1.0), cmap.N)
    peak_fields = np.argmax(z_scaled, axis=0)
    grey = (0.6, 0.6, 0.6, 0.2)

    n_fields = len(fields)
    for f_idx in range(n_fields):
        z_series = _upsample_axis(z_scaled[f_idx, :], factor=4)
        x_vals = np.linspace(1.0, float(turns), len(z_series))
        y_vals = np.full_like(x_vals, float(f_idx))
        segs = _line_segments(x_vals, y_vals, z_series)
        turn_pos = np.linspace(0, turns - 1, len(z_series))
        seg_colors = []
        for i in range(len(z_series) - 1):
            turn_idx = int(round(turn_pos[i]))
            if 0 <= turn_idx < turns and f_idx == int(peak_fields[turn_idx]):
                seg_colors.append(cmap(norm(f_idx)))
            else:
                seg_colors.append(grey)
        lc = Line3DCollection(segs, colors=seg_colors)
        lc.set_linewidth(1.8)
        ax.add_collection3d(lc)

    ax.set_xlim(1.0, float(turns))
    ax.set_ylim(0.0, float(n_fields - 1))
    ax.set_zlim(0.0, 1.0)
    ax.set_zticks([0.0, 1.0])
    try:
        ax.set_box_aspect((2.2, 1.0, 0.35))
    except Exception:
        pass

    ax.set_xlabel("Turn", labelpad=24)
    ax.set_ylabel("Focus field", labelpad=24)
    ax.set_zlabel("Score (scaled 0-1)", labelpad=24)
    ax.set_title(f"{label} lines (single conversation) - {model}", pad=20)

    tick_count = min(6, max(3, turns // 5))
    xticks = np.linspace(1, max(2, turns), tick_count, dtype=int)
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])

    field_idx = np.arange(n_fields)
    ax.set_yticks(field_idx)
    ax.set_yticklabels(fields)

    ax.tick_params(axis="x", pad=12)
    ax.tick_params(axis="y", pad=14)
    ax.tick_params(axis="z", pad=12)

    mappable = ScalarMappable(cmap=cmap, norm=norm)
    mappable.set_array([])
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.16)
    cbar.set_ticks(range(len(fields)))
    cbar.set_ticklabels(fields)
    cbar.set_label("Focus field", labelpad=10)

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
                out_dir, f"3d_lines_{label.lower()}_{model.replace(':', '_')}.png"
            )
            _plot_lines(model, turns, fields, z, label, out_path)


if __name__ == "__main__":
    main()
