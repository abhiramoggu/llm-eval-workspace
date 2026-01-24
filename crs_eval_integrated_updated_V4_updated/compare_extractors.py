#!/usr/bin/env python3
"""compare_extractors.py

Compare concept extraction methods on existing logs (no simulation needed).
Outputs:
  - figures/<mode>/extractor_compare/extractor_comparison.csv
  - figures/<mode>/extractor_compare/extractor_comparison_*.png
"""

import argparse
import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import dataset
from config import (
    CONCEPT_FIELDS,
    FIG_DIR,
    LOG_DIR,
    TAS_WEIGHTS,
    RECOVERY_WINDOW,
    ALIGNMENT_THRESHOLD,
    CONCEPT_JACCARD_SHIFT,
)
from feature_extraction import (
    CatalogFeaturizer,
    FeaturizerConfig,
    copying_ratio,
    jaccard_similarity,
    cosine_similarity,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False


YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")


def _normalize_value(field: str, value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    if not text:
        return ""
    if field == "genre":
        text = text.replace(" film", "").replace(" movie", "").replace(" fiction", "").strip()
    if field == "year":
        text = re.sub(r"[^0-9]", "", text)
    return text


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (text or "").lower())


def _candidate_values() -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for field in CONCEPT_FIELDS:
        values = list(dataset.ATTRIBUTE_CANONICAL.get(field, {}).keys())
        out[field] = [v for v in values if v]
    return out


def _extract_substring(text: str, candidates: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    grouped = {f: set() for f in CONCEPT_FIELDS}
    lowered = (text or "").lower()
    for field in CONCEPT_FIELDS:
        for norm in candidates.get(field, []):
            if norm and norm in lowered:
                grouped[field].add(f"{field}={norm}")
    return grouped


def _extract_rules(text: str, candidates: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    grouped = {f: set() for f in CONCEPT_FIELDS}
    lowered = (text or "").lower()
    if "year" in grouped:
        for y in YEAR_RE.findall(text or ""):
            if y:
                grouped["year"].add(f"year={y}")
    for field in CONCEPT_FIELDS:
        if field == "year":
            continue
        for norm in candidates.get(field, []):
            if not norm:
                continue
            pattern = r"\b" + re.escape(norm) + r"\b"
            if re.search(pattern, lowered):
                grouped[field].add(f"{field}={norm}")
    return grouped


def _fuzzy_match_tokens(candidate: str, tokens: List[str], threshold: float = 0.88) -> bool:
    if candidate in tokens:
        return True
    cand_tokens = candidate.split()
    token_set = set(tokens)
    for ct in cand_tokens:
        if ct in token_set:
            continue
        if not any(SequenceMatcher(None, ct, tok).ratio() >= threshold for tok in tokens):
            return False
    return True


def _extract_fuzzy(text: str, candidates: Dict[str, List[str]]) -> Dict[str, Set[str]]:
    grouped = {f: set() for f in CONCEPT_FIELDS}
    lowered = (text or "").lower()
    tokens = _tokenize(lowered)
    for field in CONCEPT_FIELDS:
        for norm in candidates.get(field, []):
            if not norm:
                continue
            if norm in lowered:
                grouped[field].add(f"{field}={norm}")
                continue
            if _fuzzy_match_tokens(norm, tokens):
                grouped[field].add(f"{field}={norm}")
    return grouped


def _build_embedding_index(candidates: Dict[str, List[str]]):
    if not _SKLEARN_AVAILABLE:
        return None
    index = {}
    for field, vals in candidates.items():
        if not vals:
            continue
        vect = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
        mat = vect.fit_transform(vals)
        index[field] = {"vectorizer": vect, "matrix": mat, "candidates": vals}
    return index


def _extract_embedding(text: str, embed_index, threshold: float = 0.35, top_k: int = 3) -> Dict[str, Set[str]]:
    grouped = {f: set() for f in CONCEPT_FIELDS}
    if not embed_index:
        return grouped
    for field in CONCEPT_FIELDS:
        info = embed_index.get(field)
        if not info:
            continue
        vec = info["vectorizer"].transform([text or ""])
        sims = sk_cosine(vec, info["matrix"]).ravel()
        if sims.size == 0:
            continue
        idxs = np.argsort(-sims)[:top_k]
        for idx in idxs:
            if sims[idx] >= threshold:
                norm = info["candidates"][idx]
                grouped[field].add(f"{field}={norm}")
    return grouped


def _extract_hybrid(text: str, candidates: Dict[str, List[str]], embed_index) -> Dict[str, Set[str]]:
    grouped = _extract_rules(text, candidates)
    for field in CONCEPT_FIELDS:
        if grouped[field]:
            continue
        fuzzy_grouped = _extract_fuzzy(text, {field: candidates.get(field, [])})
        grouped[field] |= fuzzy_grouped.get(field, set())
    if embed_index:
        embed_grouped = _extract_embedding(text, embed_index)
        for field in CONCEPT_FIELDS:
            if grouped[field]:
                continue
            grouped[field] |= embed_grouped.get(field, set())
    return grouped


def _flatten(grouped: Dict[str, Set[str]]) -> Set[str]:
    out: Set[str] = set()
    for vals in grouped.values():
        out |= set(vals)
    return out


def _vectorize_tokens(tokens: Set[str], vocab: Dict[str, int], idf: np.ndarray) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=float)
    for tok in tokens:
        idx = vocab.get(tok)
        if idx is None:
            continue
        vec[idx] = 1.0
    vec *= idf
    return vec


def _iter_pairs(conversation: List[Dict[str, object]]) -> List[Tuple[str, str]]:
    pairs = []
    for i in range(0, len(conversation) - 1, 2):
        a = conversation[i]
        b = conversation[i + 1]
        if a.get("speaker") == "USER" and b.get("speaker") == "SYSTEM":
            pairs.append((a.get("text", ""), b.get("text", "")))
    return pairs


def _load_logs(log_dir: str) -> List[Dict[str, object]]:
    logs = []
    for name in sorted(os.listdir(log_dir)):
        if not name.endswith(".json") or "_session_" not in name:
            continue
        path = os.path.join(log_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            logs.append(json.load(f))
    return logs

def _log_model_name(log: Dict[str, object]) -> str:
    name = log.get("model_name") or log.get("model")
    return str(name) if name else "unknown"

def _shift_points_from_log(log: Dict[str, object], conversation: List[Dict[str, object]]) -> List[int]:
    shift_points = set()
    for ev in log.get("shift_events") or []:
        if not isinstance(ev, dict):
            continue
        turn = ev.get("turn")
        if isinstance(turn, int) and turn >= 1:
            shift_points.add(turn)
    user_turn = 0
    for t in conversation:
        if t.get("speaker") != "USER":
            continue
        user_turn += 1
        meta = t.get("user_meta") or t.get("meta") or {}
        if meta.get("shift_event") or meta.get("is_shift"):
            shift_points.add(user_turn)
    return sorted(shift_points)

def _compute_recovery(shift_points: List[int], co_list: List[float], wcs_list: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not shift_points:
        return None, None
    window = max(1, int(RECOVERY_WINDOW))
    delays = []
    for sp in shift_points:
        found = None
        for d in range(0, window):
            idx = sp + d
            if idx <= len(co_list) and (co_list[idx - 1] >= CONCEPT_JACCARD_SHIFT or wcs_list[idx - 1] >= ALIGNMENT_THRESHOLD):
                found = d
                break
        if found is not None:
            delays.append(found)
    recovery_rate = float(len(delays) / len(shift_points))
    avg_delay = float(np.mean(delays)) if delays else None
    return recovery_rate, avg_delay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default=LOG_DIR)
    parser.add_argument("--out-dir", default=os.path.join(FIG_DIR, "extractor_compare"))
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-lda", action="store_true")
    parser.add_argument("--methods", help="Comma-separated extractor methods to compare")
    args = parser.parse_args()

    logs = _load_logs(args.log_dir)
    if not logs:
        raise SystemExit(f"No logs found in {args.log_dir}")

    models = sorted({_log_model_name(log) for log in logs})
    print(f"Models found: {', '.join(models)}")

    os.makedirs(args.out_dir, exist_ok=True)

    candidates = _candidate_values()
    embed_index = _build_embedding_index(candidates)
    if not _SKLEARN_AVAILABLE:
        print("Warning: sklearn not available; embedding method will be skipped.")

    base_featurizer = CatalogFeaturizer(FeaturizerConfig(topic_fields=CONCEPT_FIELDS, include_plot_keywords=False))
    vocab = base_featurizer.term2idx
    idf = base_featurizer.idf if base_featurizer.idf is not None else np.ones(len(vocab), dtype=float)

    llm_featurizer = CatalogFeaturizer(
        FeaturizerConfig(
            topic_fields=CONCEPT_FIELDS,
            include_plot_keywords=False,
            concept_extractor_mode="llm",
        )
    )
    llm_model_name = llm_featurizer.cfg.concept_extractor_model

    alpha = float(TAS_WEIGHTS.get("alpha", 1.0))
    beta = float(TAS_WEIGHTS.get("beta", 1.0))
    gamma = float(TAS_WEIGHTS.get("gamma", 1.0))

    cache: Dict[Tuple[str, str], Set[str]] = {}

    def get_tokens(method: str, text: str) -> Set[str]:
        key = (method, text)
        if key in cache:
            return cache[key]
        if method == "catalog":
            grouped = _extract_substring(text, candidates)
        elif method == "rules":
            grouped = _extract_rules(text, candidates)
        elif method == "fuzzy":
            grouped = _extract_fuzzy(text, candidates)
        elif method == "embed":
            grouped = _extract_embedding(text, embed_index)
        elif method == "hybrid":
            grouped = _extract_hybrid(text, candidates, embed_index)
        elif method == "llm":
            grouped = llm_featurizer.extract_fielded_terms(text)
        elif method == "lda":
            from evaluate import extract_topics_lda  # local import to avoid overhead
            topics = extract_topics_lda(text or "", top_k=5)
            grouped = {"topic": {f"topic={t}" for t in topics}}
        else:
            grouped = {f: set() for f in CONCEPT_FIELDS}
        tokens = _flatten(grouped)
        cache[key] = tokens
        return tokens

    supported_methods = {"catalog", "rules", "fuzzy", "embed", "hybrid", "llm", "lda"}
    methods: List[str] = []
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        env_methods = os.getenv("CONCEPT_EXTRACTOR_MODE", "")
        if env_methods:
            methods = [m.strip() for m in env_methods.split(",") if m.strip()]
    if not methods:
        methods = ["catalog", "rules", "fuzzy"]
        if _SKLEARN_AVAILABLE:
            methods += ["embed", "hybrid"]
        if not args.skip_llm:
            methods.append("llm")
        if not args.skip_lda:
            methods.append("lda")
    methods = [m for m in methods if m in supported_methods]
    if not _SKLEARN_AVAILABLE:
        methods = [m for m in methods if m not in {"embed", "hybrid"}]
    if args.skip_llm:
        methods = [m for m in methods if m != "llm"]
    if args.skip_lda:
        methods = [m for m in methods if m != "lda"]
    if not methods:
        raise SystemExit("No valid extractor methods selected.")

    # Precompute LDA vocab if needed
    lda_vocab: Dict[str, int] = {}
    if "lda" in methods:
        for log in logs:
            for u_text, s_text in _iter_pairs(log.get("conversation", [])):
                for text in (u_text, s_text):
                    toks = get_tokens("lda", text)
                    for tok in toks:
                        if tok not in lda_vocab:
                            lda_vocab[tok] = len(lda_vocab)
        if not lda_vocab:
            print("Warning: LDA produced no topics; LDA method will be empty.")

    total_pairs = 0
    for log in logs:
        total_pairs += len(_iter_pairs(log.get("conversation", [])))
    if total_pairs == 0:
        raise SystemExit("No USER/SYSTEM pairs found in logs.")

    def _progress(method: str, done: int, total: int, last_pct: int) -> int:
        pct = int((done / total) * 100) if total else 100
        if pct >= last_pct + 5 or pct == 100:
            print(f"[{method}] {pct}% ({done}/{total})")
            return pct
        return last_pct

    def _bucket():
        return {"co": [], "wcs": [], "cp": [], "tas": [], "rr": [], "rd": []}

    def _mean(vals: List[Optional[float]]) -> float:
        filtered = [v for v in vals if v is not None and not np.isnan(v)]
        return float(np.mean(filtered)) if filtered else np.nan

    def _per_log_bucket():
        return {"CO": [], "WCS": [], "CP": [], "TAS": [], "RR": [], "RD": []}

    rows = []
    per_log: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for method in methods:
        stats: Dict[str, Dict[str, List[Optional[float]]]] = {}
        done = 0
        last_pct = -5
        print(f"--- Running extractor: {method} ---")
        for log in logs:
            model = _log_model_name(log)
            conversation = log.get("conversation", [])
            pairs = _iter_pairs(conversation)
            if not pairs:
                continue
            co_list = []
            wcs_list = []
            cp_list = []
            tas_list = []
            for u_text, s_text in pairs:
                u_tokens = get_tokens(method, u_text)
                s_tokens = get_tokens(method, s_text)
                co = jaccard_similarity(u_tokens, s_tokens)
                if method == "lda":
                    vec_u = _vectorize_tokens(u_tokens, lda_vocab, np.ones(len(lda_vocab), dtype=float))
                    vec_s = _vectorize_tokens(s_tokens, lda_vocab, np.ones(len(lda_vocab), dtype=float))
                else:
                    vec_u = _vectorize_tokens(u_tokens, vocab, idf)
                    vec_s = _vectorize_tokens(s_tokens, vocab, idf)
                wcs = cosine_similarity(vec_u, vec_s)
                cp = copying_ratio(s_text, u_text)
                tas = (alpha * co) + (beta * wcs) - (gamma * cp)
                co_list.append(co)
                wcs_list.append(wcs)
                cp_list.append(cp)
                tas_list.append(tas)
                done += 1
                last_pct = _progress(method, done, total_pairs, last_pct)

            rr, rd = _compute_recovery(_shift_points_from_log(log, conversation), co_list, wcs_list)
            co_mean = float(np.mean(co_list)) if co_list else np.nan
            wcs_mean = float(np.mean(wcs_list)) if wcs_list else np.nan
            cp_mean = float(np.mean(cp_list)) if cp_list else np.nan
            tas_mean = float(np.mean(tas_list)) if tas_list else np.nan
            rr_val = float(rr) if rr is not None else np.nan
            rd_val = float(rd) if rd is not None else np.nan
            for key in (model, "__all__"):
                stats.setdefault(key, _bucket())
                stats[key]["co"].append(co_mean)
                stats[key]["wcs"].append(wcs_mean)
                stats[key]["cp"].append(cp_mean)
                stats[key]["tas"].append(tas_mean)
                stats[key]["rr"].append(rr)
                stats[key]["rd"].append(rd)
                per_log.setdefault(method, {}).setdefault(key, _per_log_bucket())
                per_log[method][key]["CO"].append(co_mean)
                per_log[method][key]["WCS"].append(wcs_mean)
                per_log[method][key]["CP"].append(cp_mean)
                per_log[method][key]["TAS"].append(tas_mean)
                per_log[method][key]["RR"].append(rr_val)
                per_log[method][key]["RD"].append(rd_val)

        for model, bucket in stats.items():
            rows.append({
                "method": method,
                "model": model,
                "metric": "CO",
                "value": _mean(bucket["co"]),
                "llm_model": llm_model_name if method == "llm" else "",
            })
            rows.append({
                "method": method,
                "model": model,
                "metric": "WCS",
                "value": _mean(bucket["wcs"]),
                "llm_model": llm_model_name if method == "llm" else "",
            })
            rows.append({
                "method": method,
                "model": model,
                "metric": "CP",
                "value": _mean(bucket["cp"]),
                "llm_model": llm_model_name if method == "llm" else "",
            })
            rows.append({
                "method": method,
                "model": model,
                "metric": "TAS",
                "value": _mean(bucket["tas"]),
                "llm_model": llm_model_name if method == "llm" else "",
            })
            rows.append({
                "method": method,
                "model": model,
                "metric": "RR",
                "value": _mean(bucket["rr"]),
                "llm_model": llm_model_name if method == "llm" else "",
            })
            rows.append({
                "method": method,
                "model": model,
                "metric": "RD",
                "value": _mean(bucket["rd"]),
                "llm_model": llm_model_name if method == "llm" else "",
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "extractor_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    metrics = ["CO", "WCS", "CP", "TAS", "RR", "RD"]
    model_labels = list(models)
    if "__all__" in df["model"].unique():
        model_labels.append("__all__")
    display_labels = ["overall" if m == "__all__" else m for m in model_labels]

    def _plot_metric_across_models(metric: str, out_path: str):
        sub = df[df["metric"] == metric].copy()
        if sub.empty:
            return
        sub = sub[sub["model"].isin(model_labels)]
        n_models = len(model_labels)
        if n_models == 0:
            return
        n_methods = max(1, len(methods))
        width = 0.8 / n_methods
        x = np.arange(n_models)
        fig_w = max(12.0, n_models * 1.2)
        plt.figure(figsize=(fig_w, 6))
        all_vals = []
        for j, method in enumerate(methods):
            vals = []
            for model in model_labels:
                row = sub[(sub["model"] == model) & (sub["method"] == method)]
                vals.append(float(row["value"].iloc[0]) if not row.empty else np.nan)
            vals = np.nan_to_num(vals, nan=0.0)
            all_vals.extend(list(vals))
            offset = (j - (n_methods - 1) / 2) * width
            plt.bar(x + offset, vals, width, label=method)
        plt.xticks(x, display_labels, rotation=25, ha="right")
        ymax = float(np.max(all_vals)) if all_vals else 0.0
        plt.ylim(0.0, max(0.05, ymax * 1.1))
        plt.ylabel("Turns" if metric == "RD" else "Score (mean over logs)")
        plt.title(f"Extractor Comparison ({metric}) — All Models")
        plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=220, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")

    for metric in metrics:
        _plot_metric_across_models(
            metric,
            os.path.join(args.out_dir, f"extractor_comparison_{metric.lower()}_all_models.png"),
        )


if __name__ == "__main__":
    main()
