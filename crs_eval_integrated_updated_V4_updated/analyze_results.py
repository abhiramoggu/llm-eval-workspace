# analyze_results.py
# Aggregates and visualizes evaluation outputs in results.jsonl.
# - Produces thesis-friendly plots with large fonts.
# - Compares TAS vs baselines (semantic similarity, DST-style set accuracy, judge rubric).
# - Plots TAS components (CO/WCS/CP only) and recovery diagnostics (RR/RD) separately.
#
# NOTE: This script is additive-only w.r.t. schemas. It reads existing results.jsonl
# and generates figures + CSV summaries. It never mutates logs/results.jsonl.

import json
import os
import shutil
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import RESULTS_FILE, FIG_DIR, CONCEPT_FIELDS

# -----------------------------
# Thesis-friendly plot defaults
# -----------------------------
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 24,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
})

SUB_DIRS = [
    "bar_charts",
    "distributions",
    "radars",
    "correlations",
    "over_time",
    "tables",
    "stacked_bars",
    "scatter",
    "three_d",
]

def _prepare_dirs(base_dir: str, sub_dirs: List[str]) -> Dict[str, str]:
    """Always clears previous visualization outputs for a fresh run."""
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    out = {}
    for s in sub_dirs:
        p = os.path.join(base_dir, s)
        os.makedirs(p, exist_ok=True)
        out[s] = p
    print(f"--- Cleared and prepared figure directories in '{base_dir}' ---")
    return out

DIRS = _prepare_dirs(FIG_DIR, SUB_DIRS)

def _load_results(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f"No records found in {path}")
    # Flatten judge dict if present
    if "judge" in df.columns:
        judge_df = pd.json_normalize(df["judge"]).add_prefix("judge.")
        df = pd.concat([df.drop(columns=["judge"]), judge_df], axis=1)
    # Ensure detail is dict
    if "detail" in df.columns:
        df["detail"] = df["detail"].apply(lambda x: x if isinstance(x, dict) else {})
    else:
        df["detail"] = [{} for _ in range(len(df))]
    return df

df = _load_results(RESULTS_FILE)

MODEL_COL = "model_name" if "model_name" in df.columns else ("model" if "model" in df.columns else None)
if MODEL_COL is None:
    raise RuntimeError("Could not find a model identifier column (expected 'model_name' or 'model').")

# -----------------------------
# Canonical metric accessors
# -----------------------------
def _col(df_: pd.DataFrame, preferred: str, fallbacks: List[str]) -> str:
    if preferred in df_.columns:
        df_[preferred] = pd.to_numeric(df_[preferred], errors="coerce")
        return preferred
    for fb in fallbacks:
        if fb in df_.columns:
            df_[fb] = pd.to_numeric(df_[fb], errors="coerce")
            return fb
    # create a zero column (safe additive) for plotting
    df_[preferred] = 0.0
    return preferred

TAS = _col(df, "trajectory_adaptation_score_mean", ["context_adaptation_score_mean", "context_adaptation_score", "adaptation_score_mean", "adaptation_score"])
CO  = _col(df, "concept_overlap_mean", ["cross_coherence_mean", "cross_coherence"])
WCS = _col(df, "weighted_constraint_similarity_mean", ["constraint_similarity_mean", "context_retention_mean", "context_retention"])
CP  = _col(df, "copying_penalty_mean", ["copy_penalty_mean", "topic_interference_mean", "topic_interference"])

RR  = _col(df, "topic_recovery_rate", ["recovery_rate"])
RD  = _col(df, "avg_recovery_delay", ["avg_delay"])

RR_COS = _col(df, "recovery_rate_cosine", [])
RD_COS = _col(df, "avg_recovery_delay_cosine", [])

SEM = _col(df, "semantic_similarity_mean", [])
DST_F1 = _col(df, "dst_f1_mean", [])

# Judge metrics (1..5 typically)
judge_cols = [c for c in df.columns if c.startswith("judge.rubric_")]
if not judge_cols:
    # compatibility: older keys without prefix
    judge_cols = [c for c in df.columns if c.startswith("rubric_")]

# -----------------------------
# Summary tables
# -----------------------------
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_summary = df.groupby(MODEL_COL)[num_cols].mean().reset_index()

summary_csv = os.path.join(DIRS["tables"], "model_metrics.csv")
df_summary.to_csv(summary_csv, index=False)
print(f"Saved: {summary_csv}")

# Recovery table
df_recovery = df_summary[[MODEL_COL, RR, RD, RR_COS, RD_COS]].copy()
df_recovery.to_csv(os.path.join(DIRS["tables"], "recovery_metrics.csv"), index=False)

# -----------------------------
# Per-field concept diagnostics
# -----------------------------
def _iter_concept_tokens(obj):
    if obj is None:
        return
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_concept_tokens(item)
    elif isinstance(obj, str):
        yield obj

def _token_field(token: str) -> Optional[str]:
    if not isinstance(token, str) or "=" not in token:
        return None
    return token.split("=", 1)[0]

def _accumulate_field_counters(df_: pd.DataFrame, col: str):
    overall = {f: Counter() for f in CONCEPT_FIELDS}
    by_model = {f: {} for f in CONCEPT_FIELDS}
    for _, row in df_.iterrows():
        model = row[MODEL_COL]
        tokens = row.get(col)
        if tokens is None:
            continue
        for tok in _iter_concept_tokens(tokens):
            field = _token_field(tok)
            if field in overall:
                overall[field][tok] += 1
                by_model.setdefault(field, {})
                by_model[field].setdefault(model, Counter())
                by_model[field][model][tok] += 1
    return overall, by_model

missing_overall, missing_by_model = _accumulate_field_counters(df, "missing_concepts_per_turn")
hall_overall, hall_by_model = _accumulate_field_counters(df, "hallucinated_concepts_per_turn")

for field in CONCEPT_FIELDS:
    overall_counter = missing_overall.get(field, Counter())
    if overall_counter:
        pd.DataFrame(overall_counter.most_common(200), columns=["concept", "count"]).to_csv(
            os.path.join(DIRS["tables"], f"top_missing_by_field_{field}.csv"), index=False
        )
    rows = []
    for model, counter in (missing_by_model.get(field, {}) or {}).items():
        for concept, cnt in counter.most_common(100):
            rows.append({MODEL_COL: model, "concept": concept, "count": cnt})
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(DIRS["tables"], f"top_missing_by_field_{field}_by_model.csv"), index=False
        )

    overall_counter = hall_overall.get(field, Counter())
    if overall_counter:
        pd.DataFrame(overall_counter.most_common(200), columns=["concept", "count"]).to_csv(
            os.path.join(DIRS["tables"], f"top_hallucinated_by_field_{field}.csv"), index=False
        )
    rows = []
    for model, counter in (hall_by_model.get(field, {}) or {}).items():
        for concept, cnt in counter.most_common(100):
            rows.append({MODEL_COL: model, "concept": concept, "count": cnt})
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(DIRS["tables"], f"top_hallucinated_by_field_{field}_by_model.csv"), index=False
        )

# Debug artifact: sample turns with per-field concepts
def _collect_debug_samples(df_: pd.DataFrame, max_samples: int = 20) -> List[dict]:
    samples: List[dict] = []
    for _, row in df_.iterrows():
        model = row[MODEL_COL]
        detail = row.get("detail") or {}
        for t in detail.get("turn_level", []) or []:
            if not isinstance(t, dict):
                continue
            if "user_text" not in t or "system_text" not in t:
                continue
            samples.append({
                "model": model,
                "turn": t.get("turn"),
                "user_text": t.get("user_text", ""),
                "system_text": t.get("system_text", ""),
                "user_concepts_by_field": t.get("user_concepts_by_field", {}),
                "system_concepts_by_field": t.get("system_concepts_by_field", {}),
                "missing_by_field": t.get("missing_by_field", {}),
                "hallucinated_by_field": t.get("hallucinated_by_field", {}),
            })
    if len(samples) <= max_samples:
        return samples
    rng = np.random.default_rng(0)
    idxs = rng.choice(len(samples), size=max_samples, replace=False)
    return [samples[i] for i in idxs]

debug_samples = _collect_debug_samples(df, max_samples=20)
if debug_samples:
    debug_path = os.path.join(DIRS["tables"], "concept_debug_samples.jsonl")
    with open(debug_path, "w", encoding="utf-8") as f:
        for row in debug_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved: {debug_path}")

# -----------------------------
# Helper plotting functions
# -----------------------------
def _legend_outside():
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

def _save(figpath: str):
    fig = plt.gcf()
    has_3d = any(getattr(ax, "name", "") == "3d" for ax in fig.axes)
    if has_3d:
        fig.subplots_adjust(left=0.08, right=0.9, top=0.92, bottom=0.12)
    else:
        fig.tight_layout()
    plt.savefig(figpath, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {figpath}")

def _set_bar_ylim(values: np.ndarray, floor: float = 0.0, pad: float = 0.1, min_top: float = 0.05):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    vmax = float(np.max(vals)) if vals.size else 0.0
    if vmax <= 0.0:
        vmax = min_top
    top = max(min_top, vmax * (1.0 + pad))
    plt.ylim(floor, top)

def _kde_curve(values: np.ndarray, num: int = 200, bandwidth: Optional[float] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None, None
    std = float(np.std(vals))
    if std == 0.0:
        return None, None
    if bandwidth is None:
        bandwidth = 1.06 * std * (vals.size ** (-1.0 / 5.0))
    if bandwidth <= 0.0:
        return None, None
    grid = np.linspace(float(vals.min()), float(vals.max()), num=num)
    diff = (grid[:, None] - vals[None, :]) / bandwidth
    dens = np.exp(-0.5 * diff ** 2).sum(axis=1) / (vals.size * bandwidth * np.sqrt(2 * np.pi))
    return grid, dens

def _pretty_metric_name(name: str) -> str:
    return name.replace("judge.", "").replace("_", " ").title()

# -----------------------------
# 1) Bar chart: TAS vs baselines
# -----------------------------
base_metrics = [TAS, SEM, DST_F1]
labels = {
    TAS: "TAS",
    SEM: "Semantic Similarity",
    DST_F1: "DST F1 (set overlap)",
}
plot_df = df_summary[[MODEL_COL] + base_metrics].copy()

x = np.arange(len(plot_df[MODEL_COL]))
width = 0.18
plt.figure(figsize=(16, 7))
for j, mcol in enumerate(base_metrics):
    plt.bar(x + (j - 1.5) * width, plot_df[mcol].values, width, label=labels[mcol])
plt.xticks(x, plot_df[MODEL_COL], rotation=30, ha="right")
_set_bar_ylim(plot_df[base_metrics].to_numpy().flatten(), floor=0.0)
plt.ylabel("Score (0–1)")
plt.title("Proposed TAS vs Baselines (mean across conversations)")
_legend_outside()
_save(os.path.join(DIRS["bar_charts"], "tas_vs_baselines.png"))

# -----------------------------
# 2) Bar chart: TAS components only
# -----------------------------
comp_df = df_summary[[MODEL_COL, CO, WCS, CP]].copy()
comp_df["1-CP"] = 1.0 - comp_df[CP].astype(float)

plt.figure(figsize=(16, 7))
metrics2 = [CO, WCS, CP]
width = 0.22
for j, mcol in enumerate(metrics2):
    plt.bar(x + (j - 1) * width, comp_df[mcol].values, width, label=mcol)
plt.xticks(x, comp_df[MODEL_COL], rotation=30, ha="right")
_set_bar_ylim(comp_df[metrics2].to_numpy().flatten(), floor=0.0)
plt.ylabel("Component score (0–1)")
plt.title("TAS Components by Model")
_legend_outside()
_save(os.path.join(DIRS["bar_charts"], "tas_components_bar.png"))

# -----------------------------
# 3) TAS components radar (NO CAS)
# -----------------------------
labels_r = ["CO", "WCS", "1-CP"]
angles = np.linspace(0, 2*np.pi, len(labels_r), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)
for _, row in comp_df.iterrows():
    vals = [float(row[CO]), float(row[WCS]), float(row["1-CP"])]
    vals = [float(np.clip(v, 0.0, 1.0)) for v in vals]
    vals += vals[:1]
    ax.plot(angles, vals, linewidth=2, label=row[MODEL_COL])
    ax.fill(angles, vals, alpha=0.08)
ax.set_thetagrids(np.degrees(angles[:-1]), labels_r)
ax.set_title("TAS Components Radar (higher is better)", pad=25)
ax.legend(loc="center left", bbox_to_anchor=(1.15, 0.5), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(DIRS["radars"], "tas_components_radar.png"), dpi=220, bbox_inches="tight")
plt.close()

# -----------------------------
# 4) Recovery RR/RD plots
# -----------------------------
plt.figure(figsize=(16, 6))
plt.bar(x - width/2, df_recovery[RR].fillna(0).values, width, label="RR (Jaccard)")
plt.bar(x + width/2, df_recovery[RR_COS].fillna(0).values, width, label="RR (Cosine)")
plt.xticks(x, df_recovery[MODEL_COL], rotation=30, ha="right")
_set_bar_ylim(df_recovery[[RR, RR_COS]].to_numpy().flatten(), floor=0.0)
plt.ylabel("Recovery Rate")
plt.title("Shift Recovery Rate")
_legend_outside()
_save(os.path.join(DIRS["bar_charts"], "recovery_rate.png"))

plt.figure(figsize=(16, 6))
plt.bar(x - width/2, df_recovery[RD].fillna(0).values, width, label="RD (Jaccard)")
plt.bar(x + width/2, df_recovery[RD_COS].fillna(0).values, width, label="RD (Cosine)")
plt.xticks(x, df_recovery[MODEL_COL], rotation=30, ha="right")
_set_bar_ylim(df_recovery[[RD, RD_COS]].fillna(0).to_numpy().flatten(), floor=0.0, min_top=0.5)
plt.ylabel("Avg Recovery Delay (turns)")
plt.title("Shift Recovery Delay")
_legend_outside()
_save(os.path.join(DIRS["bar_charts"], "recovery_delay.png"))

# -----------------------------
# 5) Judge stacked normalized (0–1)
# -----------------------------
if judge_cols:
    jdf = df_summary[[MODEL_COL] + judge_cols].copy()
    # normalize [1,5] -> [0,1]
    for c in judge_cols:
        jdf[c] = (jdf[c].astype(float) - 1.0) / 4.0

    plt.figure(figsize=(16, 7))
    bottom = np.zeros(len(jdf))
    for c in judge_cols:
        plt.bar(x, jdf[c].values, bottom=bottom, label=c.split(".")[-1].replace("_", " ").title())
        bottom += jdf[c].values
    plt.xticks(x, jdf[MODEL_COL], rotation=30, ha="right")
    plt.ylabel("Normalized judge score (stacked)")
    plt.title("LLM-as-a-Judge Rubric (Normalized 0–1)")
    _legend_outside()
    _save(os.path.join(DIRS["stacked_bars"], "judge_stacked_normalized.png"))

# -----------------------------
# 6) Distributions (TAS + baselines)
# -----------------------------
for mcol, name in [(TAS, "TAS"), (SEM, "Semantic Similarity"), (DST_F1, "DST F1")]:
    plt.figure(figsize=(12, 6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, model in enumerate(df[MODEL_COL].unique()):
        vals = df[df[MODEL_COL] == model][mcol].dropna().astype(float).values
        if len(vals) == 0:
            continue
        color = colors[idx % len(colors)] if colors else None
        if np.std(vals) == 0.0:
            plt.axvline(float(vals[0]), linestyle="--", color=color, label=f"{model} (single value)")
            continue
        xs, ys = _kde_curve(vals)
        if xs is None or ys is None:
            continue
        plt.plot(xs, ys, linewidth=2, color=color, label=model)
        plt.fill_between(xs, ys, alpha=0.2, color=color)
    plt.title(f"Distribution: {name}")
    plt.xlabel("Score")
    plt.ylabel("Density (KDE)")
    _legend_outside()
    _save(os.path.join(DIRS["distributions"], f"dist_{mcol}.png"))

# -----------------------------
# 7) Correlations (TAS vs baselines, TAS vs judge) + pairwise scatter
# -----------------------------
corr_cols = [TAS, SEM, DST_F1]
if judge_cols:
    corr_cols += judge_cols
corr_df = df[corr_cols].copy()
for c in corr_cols:
    corr_df[c] = pd.to_numeric(corr_df[c], errors="coerce")
corr = corr_df.corr(method="spearman")
corr.to_csv(os.path.join(DIRS["tables"], "spearman_correlation.csv"))

colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
for i in range(len(corr_cols)):
    for j in range(i + 1, len(corr_cols)):
        x_col = corr_cols[i]
        y_col = corr_cols[j]
        rho = corr.loc[x_col, y_col]
        plt.figure(figsize=(10, 8))
        for idx, model in enumerate(df[MODEL_COL].unique()):
            sub = df[df[MODEL_COL] == model]
            x = pd.to_numeric(sub[x_col], errors="coerce")
            y = pd.to_numeric(sub[y_col], errors="coerce")
            mask = x.notna() & y.notna()
            if mask.sum() == 0:
                continue
            color = colors[idx % len(colors)] if colors else None
            plt.scatter(x[mask], y[mask], alpha=0.55, s=40, color=color, label=model)
            plt.scatter([x[mask].mean()], [y[mask].mean()], marker="X", s=120, color=color, edgecolors="black", linewidths=0.5)
            if mask.sum() >= 2 and np.unique(x[mask]).size >= 2:
                coeffs = np.polyfit(x[mask], y[mask], 1)
                xs = np.linspace(float(np.min(x[mask])), float(np.max(x[mask])), num=50)
                ys = (coeffs[0] * xs) + coeffs[1]
                plt.plot(xs, ys, color=color, linewidth=1.5, alpha=0.8)
        plt.xlabel(_pretty_metric_name(x_col))
        plt.ylabel(_pretty_metric_name(y_col))
        title = f"{_pretty_metric_name(x_col)} vs {_pretty_metric_name(y_col)}"
        if pd.notna(rho):
            title += f" (Spearman ρ={rho:.3f})"
        plt.title(title)
        _legend_outside()
        _save(os.path.join(DIRS["correlations"], f"scatter_{x_col}_vs_{y_col}.png"))

# -----------------------------
# 8) Over-time: TAS per turn (mean across conversations)
# -----------------------------
def _extract_turn_series(detail: dict, key: str) -> List[float]:
    # grounded logs use detail.turn_scores.{tas, cc, cr, i}
    ts = detail.get("turn_scores") or {}
    series = ts.get(key)
    if isinstance(series, list):
        return [float(x) for x in series]
    tl = detail.get("turn_level") or []
    out: List[float] = []
    if isinstance(tl, list):
        for row in tl:
            if not isinstance(row, dict):
                continue
            if key not in row:
                continue
            val = row.get(key)
            if val is None:
                continue
            try:
                out.append(float(val))
            except (TypeError, ValueError):
                continue
    return out

series_by_model: Dict[str, List[List[float]]] = {}
for _, row in df.iterrows():
    series = _extract_turn_series(row["detail"], "tas")
    if not series:
        continue
    series_by_model.setdefault(row[MODEL_COL], []).append(series)

if series_by_model:
    max_len = max(len(s) for arr in series_by_model.values() for s in arr)
    plt.figure(figsize=(16, 7))
    for model, arr in series_by_model.items():
        mat = np.full((len(arr), max_len), np.nan, dtype=float)
        for i, s in enumerate(arr):
            mat[i, :len(s)] = s
        mean = np.nanmean(mat, axis=0)
        plt.plot(np.arange(1, max_len + 1), mean, linewidth=3, label=model)
    plt.xlabel("Turn index")
    plt.ylabel("TAS (mean)")
    plt.title("TAS over Turns (mean across conversations)")
    _legend_outside()
    _save(os.path.join(DIRS["over_time"], "tas_over_turns.png"))

# -----------------------------
# 9) Conversation deviation / volatility (remedy 'nothing shown')
# -----------------------------
# Volatility = stddev of per-turn TAS within a conversation; then mean per model.
vol_rows = []
for _, row in df.iterrows():
    series = _extract_turn_series(row["detail"], "tas")
    if series:
        vol_rows.append({MODEL_COL: row[MODEL_COL], "tas_volatility": float(np.std(series))})
if vol_rows:
    vol_df = pd.DataFrame(vol_rows)
    vol_summary = vol_df.groupby(MODEL_COL)["tas_volatility"].mean().reset_index()
    vol_summary.to_csv(os.path.join(DIRS["tables"], "tas_volatility.csv"), index=False)

    plt.figure(figsize=(16, 6))
    models = vol_summary[MODEL_COL].tolist()
    data = [vol_df[vol_df[MODEL_COL] == m]["tas_volatility"].values for m in models]
    plt.boxplot(data, tick_labels=models, showmeans=True, meanline=True)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Stddev(TAS per turn)")
    plt.title("Conversation Deviation (TAS volatility)")
    _save(os.path.join(DIRS["bar_charts"], "tas_volatility.png"))

# -----------------------------
# 10) 3D visualization (turn × component × value)
# -----------------------------
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    def _extract_turn_components(detail: dict) -> Optional[Dict[str, List[float]]]:
        tl = detail.get("turn_level") or []
        if not isinstance(tl, list) or not tl:
            return None
        out = {"cc": [], "cr": [], "i": [], "tas": [], "focus": []}
        for row in tl:
            if not isinstance(row, dict):
                continue
            try:
                cc = float(row.get("cc"))
                cr = float(row.get("cr"))
                i = float(row.get("i"))
                tas = float(row.get("tas"))
            except (TypeError, ValueError):
                continue
            meta = row.get("user_meta") or {}
            focus = meta.get("focus_field") if isinstance(meta, dict) else None
            out["cc"].append(cc)
            out["cr"].append(cr)
            out["i"].append(i)
            out["tas"].append(tas)
            out["focus"].append(str(focus) if focus else "")
        if not out["tas"]:
            return None
        return out

    def _mean_series(series_list: List[List[float]], max_len: int) -> np.ndarray:
        mat = np.full((len(series_list), max_len), np.nan, dtype=float)
        for i, s in enumerate(series_list):
            mat[i, :len(s)] = np.asarray(s, dtype=float)
        return np.nanmean(mat, axis=0)

    series_by_model = {}
    for _, row in df.iterrows():
        detail = row["detail"]
        model = row[MODEL_COL]
        comps = _extract_turn_components(detail)
        if not comps:
            continue
        series_by_model.setdefault(model, {"cc": [], "cr": [], "i": [], "tas": [], "focus": []})
        for key in ["cc", "cr", "i", "tas", "focus"]:
            series_by_model[model][key].append(comps[key])

    for model, comp_series in series_by_model.items():
        if not comp_series["tas"]:
            continue
        max_len = max(len(s) for s in comp_series["tas"])
        turns = np.arange(1, max_len + 1)
        means = {k: _mean_series(comp_series[k], max_len) for k in ["cc", "cr", "i", "tas"]}
        comps = [("CO", "cc"), ("WCS", "cr"), ("CP", "i"), ("TAS", "tas")]
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection="3d")
        for ci, (name, key) in enumerate(comps):
            z = np.asarray(means[key], dtype=float)
            y = np.full_like(turns, ci, dtype=float)
            ax.plot(turns, y, z, marker="o", linewidth=1.2, markersize=3, label=name)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Component")
        ax.set_yticks(range(len(comps)))
        ax.set_yticklabels([c[0] for c in comps])
        ax.set_zlabel("Score")
        ax.set_title(f"3D: TAS components over turns ({model})")
        ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(DIRS["three_d"], f"3d_tas_components_{model}.png"), dpi=220, bbox_inches="tight")
        plt.close()

        def _plot_score_sheet(series_list: List[List[float]], focus_list: List[List[str]], label: str, filename: str):
            if not series_list or not focus_list:
                return
            max_len = max(len(s) for s in series_list)
            if max_len == 0:
                return
            fields = list(CONCEPT_FIELDS)
            fields.append("other")
            field_to_idx = {f: i for i, f in enumerate(fields)}
            sum_mat = np.zeros((len(fields), max_len), dtype=float)
            count_mat = np.zeros((len(fields), max_len), dtype=float)
            for s_idx, s in enumerate(series_list):
                f_series = focus_list[s_idx] if s_idx < len(focus_list) else []
                for t_idx, score in enumerate(s):
                    if t_idx >= max_len:
                        continue
                    focus = f_series[t_idx] if t_idx < len(f_series) else ""
                    field = focus if focus in field_to_idx else "other"
                    y_idx = field_to_idx[field]
                    sum_mat[y_idx, t_idx] += float(score)
                    count_mat[y_idx, t_idx] += 1.0
            with np.errstate(invalid="ignore", divide="ignore"):
                mat = np.divide(sum_mat, count_mat)
                mat[count_mat == 0] = np.nan
            turns = np.arange(1, max_len + 1)
            field_idx = np.arange(len(fields))
            X, Y = np.meshgrid(turns, field_idx)
            Z = mat
            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
            ax.set_xlabel("Turn")
            ax.set_ylabel("Focus field")
            ax.set_yticks(field_idx)
            ax.set_yticklabels(fields)
            ax.set_zlabel("Score")
            ax.set_title(f"{label} sheet by focus field ({model})")
            plt.tight_layout()
            plt.savefig(os.path.join(DIRS["three_d"], filename), dpi=220, bbox_inches="tight")
            plt.close()

        _plot_score_sheet(comp_series["tas"], comp_series["focus"], "TAS", f"3d_tas_sheet_{model}.png")
        _plot_score_sheet(comp_series["cc"], comp_series["focus"], "CO", f"3d_co_sheet_{model}.png")
        _plot_score_sheet(comp_series["cr"], comp_series["focus"], "WCS", f"3d_wcs_sheet_{model}.png")

    def _collect_field_count_series(detail: dict, key: str) -> List[List[int]]:
        tl = detail.get("turn_level") or []
        out: List[List[int]] = []
        if not isinstance(tl, list):
            return out
        for row in tl:
            if not isinstance(row, dict):
                continue
            mapping = row.get(key) or {}
            if not isinstance(mapping, dict):
                mapping = {}
            out.append([len(mapping.get(f, [])) for f in CONCEPT_FIELDS])
        return out

    def _mean_field_surface(series_list: List[List[List[int]]], field_count: int) -> Tuple[Optional[np.ndarray], int]:
        if not series_list:
            return None, 0
        max_len = max(len(s) for s in series_list)
        if max_len == 0:
            return None, 0
        mat = np.full((len(series_list), field_count, max_len), np.nan, dtype=float)
        for i, s in enumerate(series_list):
            for t_idx, counts in enumerate(s):
                if t_idx >= max_len or len(counts) != field_count:
                    continue
                mat[i, :, t_idx] = np.asarray(counts, dtype=float)
        return np.nanmean(mat, axis=0), max_len

    field_names = list(CONCEPT_FIELDS)
    field_series_by_model: Dict[str, Dict[str, List[List[List[int]]]]] = {}
    for _, row in df.iterrows():
        detail = row["detail"]
        model = row[MODEL_COL]
        user_series = _collect_field_count_series(detail, "user_concepts_by_field")
        sys_series = _collect_field_count_series(detail, "system_concepts_by_field")
        if user_series:
            field_series_by_model.setdefault(model, {"user": [], "system": []})
            field_series_by_model[model]["user"].append(user_series)
        if sys_series:
            field_series_by_model.setdefault(model, {"user": [], "system": []})
            field_series_by_model[model]["system"].append(sys_series)

    for model, buckets in field_series_by_model.items():
        for label, series_list in buckets.items():
            surface, max_len = _mean_field_surface(series_list, len(field_names))
            if surface is None or max_len == 0:
                continue
            turns = np.arange(1, max_len + 1)
            fields_idx = np.arange(len(field_names))
            X, Y = np.meshgrid(turns, fields_idx)
            Z = surface

            fig = plt.figure(figsize=(12, 9))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.85, linewidth=0, antialiased=True)
            ax.set_xlabel("Turn")
            ax.set_ylabel("Field")
            ax.set_yticks(fields_idx)
            ax.set_yticklabels(field_names)
            ax.set_zlabel("Concept count")
            ax.set_title(f"Concept frequency sheet ({label.upper()}) - {model}")
            plt.tight_layout()
            plt.savefig(os.path.join(DIRS["three_d"], f"concept_field_sheet_{label}_{model}.png"), dpi=220, bbox_inches="tight")
            plt.close()
except Exception:
    pass

print("--- Analysis complete. ---")
