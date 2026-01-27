#!/usr/bin/env python3
"""Generate single-conversation 3D sheets per model for TAS and components.

Outputs are written to figures/<mode>/3D_test_run_0126 without touching
existing logs or figures.
"""

import json
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from config import RESULTS_FILE, FIG_DIR, CONCEPT_FIELDS


RNG_SEED = 126
OUT_SUBDIR = "3D_test_run_0126"
SWAP_AXES = bool(int(os.getenv("SWAP_AXES", "0")))


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


def _upsample_grid(z: np.ndarray, x_factor: int = 4, y_factor: int = 3) -> np.ndarray:
    if z.size == 0:
        return np.array([])
    # Upsample along X (turns)
    up_x = np.stack([_upsample_axis(row, x_factor) for row in z], axis=0)
    # Upsample along Y (fields)
    if y_factor <= 1 or up_x.shape[0] < 2:
        up_y = up_x
    else:
        y_old = np.linspace(0.0, 1.0, up_x.shape[0])
        y_new = np.linspace(0.0, 1.0, (up_x.shape[0] - 1) * y_factor + 1)
        up_y = np.stack([np.interp(y_new, y_old, up_x[:, i]) for i in range(up_x.shape[1])], axis=1)
    return up_y


def _normalize_unit(z: np.ndarray) -> np.ndarray:
    finite = z[np.isfinite(z)]
    if finite.size == 0:
        return z
    z_min = float(np.min(finite))
    z_max = float(np.max(finite))
    if z_max <= z_min:
        return np.zeros_like(z)
    return (z - z_min) / (z_max - z_min)


def _smooth_surface(z: np.ndarray, passes: int = 3) -> np.ndarray:
    kernel = np.array([1, 4, 6, 4, 1], dtype=float)
    kernel = kernel / np.sum(kernel)
    out = z.copy()
    for _ in range(passes):
        tmp = np.zeros_like(out)
        for i in range(out.shape[0]):
            tmp[i, :] = np.convolve(out[i, :], kernel, mode="same")
        out2 = np.zeros_like(out)
        for j in range(out.shape[1]):
            out2[:, j] = np.convolve(tmp[:, j], kernel, mode="same")
        out = out2
    return out


def _plot_sheet(model: str, turns: int, fields: List[str], z: np.ndarray, label: str, out_path: str):
    if z.size == 0:
        return
    z_scaled = _normalize_unit(z)
    z_scaled = _smooth_along_turns(z_scaled)
    z_scaled = _normalize_unit(z_scaled)
    z_up = _upsample_grid(z_scaled, x_factor=8, y_factor=6)
    if z_up.size == 0:
        return
    z_up = _smooth_surface(z_up, passes=6)
    z_up = _normalize_unit(z_up)
    if SWAP_AXES:
        z_plot = z_up.T
        x_vals = np.linspace(1.0, float(turns), z_plot.shape[0])
        y_vals = np.linspace(0.0, float(len(fields) - 1), z_plot.shape[1])
        x_label = "Turn"
        y_label = "Focus field"
    else:
        z_plot = z_up
        x_vals = np.linspace(0.0, float(len(fields) - 1), z_plot.shape[0])
        y_vals = np.linspace(1.0, float(turns), z_plot.shape[1])
        x_label = "Focus field"
        y_label = "Turn"
    X, Y = np.meshgrid(x_vals, y_vals, indexing="ij")

    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, z_plot, cmap="viridis", linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel(x_label, labelpad=24)
    ax.set_ylabel(y_label, labelpad=24)
    ax.set_zlabel("Score (scaled 0-1)", labelpad=24)
    ax.set_title(f"{label} sheet (single conversation) - {model}", pad=20)
    ax.set_zlim(0.0, 1.0)
    ax.set_zticks([0.0, 1.0])
    try:
        if SWAP_AXES:
            ax.set_box_aspect((2.2, 1.0, 0.35))
        else:
            ax.set_box_aspect((1.0, 2.2, 0.35))
    except Exception:
        pass

    field_idx = np.arange(len(fields))
    tick_count = min(6, max(3, turns // 5))
    turn_ticks = np.linspace(1, max(2, turns), tick_count, dtype=int)
    if SWAP_AXES:
        ax.set_xticks(turn_ticks)
        ax.set_xticklabels([str(t) for t in turn_ticks])
        ax.set_yticks(field_idx)
        ax.set_yticklabels(fields)
    else:
        ax.set_xticks(field_idx)
        ax.set_xticklabels(fields)
        ax.set_yticks(turn_ticks)
        ax.set_yticklabels([str(t) for t in turn_ticks])

    ax.tick_params(axis="x", pad=12)
    ax.tick_params(axis="y", pad=14)
    ax.tick_params(axis="z", pad=12)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.16)
    cbar.set_label("Score (scaled 0-1)", labelpad=10)

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
        focus_idx_by_turn = []
        for row in tl:
            meta = row.get("user_meta") if isinstance(row, dict) else {}
            focus = meta.get("focus_field") if isinstance(meta, dict) else None
            focus_field = focus if focus in field_to_idx else "other"
            focus_idx_by_turn.append(field_to_idx[focus_field])
        for label, key in metrics.items():
            z = np.zeros((len(fields), turns), dtype=float)
            for t_idx, row in enumerate(tl):
                if not isinstance(row, dict):
                    continue
                try:
                    score = float(row.get(key))
                except (TypeError, ValueError):
                    score = 0.0
                weights = _field_weights(focus_idx_by_turn[t_idx], len(fields))
                z[:, t_idx] = score * weights

            out_path = os.path.join(
                out_dir, f"3d_sheet_{label.lower()}_{model.replace(':', '_')}.png"
            )
            _plot_sheet(model, turns, fields, z, label, out_path)


if __name__ == "__main__":
    main()
