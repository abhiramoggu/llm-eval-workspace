# feature_extraction.py
# Catalog-grounded feature extraction for CRS evaluation.
#
# This project evaluates CRS adaptation under a CLOSED catalog. To make the
# evaluation grounded and interpretable, we extract features ONLY from
# catalog-defined vocabulary.
#
# Representations
# ---------------
# 1) Concept set (grounded constraints) (fielded):
#    concept_set(x) = {f"{field}={normalized_value}" ...}
#    -> used with Jaccard similarity for inter-topic alignment.
#
# 2) Attribute vector (idf-weighted, fielded):
#    vec(x)[i] = idf(term_i) if term_i mentioned in x else 0
#    -> used with cosine similarity for intra-topic alignment.
#
# Plot/style descriptors are handled via a controlled keyword list extracted
# from plot summaries (dataset.PLOT_KEYWORDS). These are optional and disabled
# by default for TAS/diagnostics.
#
# NOTE: This module intentionally avoids any "open" text embedding that could
# drift away from the catalog grounding, because the thesis aims to show a
# grounded alternative to LLM-as-judge scoring.

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

import dataset
from config import (
    CONCEPT_EXTRACTOR_MODE,
    CONCEPT_EXTRACTOR_MODEL,
    CONCEPT_EMBED_THRESHOLD,
    CONCEPT_EMBED_TOP_K,
    CONCEPT_FIELDS,
    PLOT_FIELD,
    TITLE_FIELD,
    USE_PLOT_KW_FOR_TAS,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as _sk_cosine_similarity
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

_EMBED_WARNED = False


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

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

def _canonicalize_value(field: str, value: str) -> Optional[str]:
    norm = _normalize_value(field, value)
    if not norm:
        return None
    candidates = dataset.ATTRIBUTE_CANONICAL.get(field, {})
    if norm in candidates:
        return norm
    for cand_norm in candidates:
        if norm in cand_norm or cand_norm in norm:
            return cand_norm
    return None

def _extract_json(text: str) -> str:
    if not text:
        return ""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return text[start : end + 1]

def _call_ollama(model: str, prompt: str) -> str:
    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return ""
    if proc.returncode != 0:
        return ""
    err = (proc.stderr or "").strip().lower()
    if err and "error" in err:
        return ""
    return (proc.stdout or "").strip()


def word_tokenize(text: str) -> List[str]:
    """Lowercased, simple word tokenizer for overlap-based metrics."""
    if not text:
        return []
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def ngrams(tokens: Sequence[str], n: int) -> Set[Tuple[str, ...]]:
    if n <= 0:
        return set()
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(0, len(tokens) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    if u is None or v is None:
        return 0.0
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def copy_ratio(system_text: str, user_text: str, n_orders: Sequence[int] = (3, 4)) -> float:
    """Max n-gram copying ratio: |G_n(sys) ∩ G_n(user)| / |G_n(sys)|."""
    sys_toks = word_tokenize(system_text)
    user_toks = word_tokenize(user_text)
    if not sys_toks:
        return 0.0
    ratios = []
    for n in n_orders:
        gs = ngrams(sys_toks, n)
        if not gs:
            continue
        gu = ngrams(user_toks, n)
        ratios.append(len(gs & gu) / len(gs))
    return max(ratios) if ratios else 0.0


# Backward-compatible aliases (legacy naming in docs/evaluate.py).
def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    return jaccard(a, b)


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    return cosine(u, v)


def copying_ratio(system_text: str, user_text: str, n_orders: Sequence[int] = (3, 4)) -> float:
    return copy_ratio(system_text, user_text, n_orders=n_orders)


def ngram_copy_ratio(system_text: str, user_text: str, n_orders: Sequence[int] = (3, 4)) -> float:
    return copy_ratio(system_text, user_text, n_orders=n_orders)


@dataclass(frozen=True)
class FeaturizerConfig:
    # NOTE: legacy name "topic_fields" == "concept_fields" in the thesis.
    topic_fields: Tuple[str, ...] = CONCEPT_FIELDS
    include_plot_keywords: bool = USE_PLOT_KW_FOR_TAS
    plot_field: str = PLOT_FIELD
    concept_extractor_mode: str = CONCEPT_EXTRACTOR_MODE
    concept_extractor_model: str = CONCEPT_EXTRACTOR_MODEL
    embed_threshold: float = CONCEPT_EMBED_THRESHOLD
    embed_top_k: int = CONCEPT_EMBED_TOP_K


class CatalogFeaturizer:
    """Builds catalog-grounded topic sets and attribute vectors."""

    def __init__(self, cfg: Optional[FeaturizerConfig] = None):
        self.cfg = cfg or FeaturizerConfig()
        self._embed_index = None
        self._embed_threshold = float(self.cfg.embed_threshold)
        self._embed_top_k = max(1, int(self.cfg.embed_top_k))
        if self.cfg.concept_extractor_mode == "embed":
            self._embed_index = self._build_embed_index()
            if self._embed_index is None:
                global _EMBED_WARNED
                if not _EMBED_WARNED:
                    print("Warning: sklearn not available; falling back to heuristic extractor.")
                    _EMBED_WARNED = True

        # Vocabulary = fielded terms for structured fields + plot keywords
        self.term2idx: Dict[str, int] = {}
        self.idx2term: List[str] = []
        self.idf: Optional[np.ndarray] = None

        self._build_vocab_and_idf()

    def _add_term(self, term: str):
        if term not in self.term2idx:
            self.term2idx[term] = len(self.idx2term)
            self.idx2term.append(term)

    def _build_vocab_and_idf(self):
        # Document frequency (df) uses number of catalog items containing term.
        # This is a stable, grounded alternative to fitting TF-IDF on arbitrary
        # conversation text.
        N = max(1, len(dataset.MOVIE_DB))
        df: Dict[str, int] = {}

        for field in self.cfg.topic_fields:
            if field not in dataset.ATTRIBUTE_INDEX:
                continue
            for norm_val, titles in dataset.ATTRIBUTE_INDEX[field].items():
                term = f"{field}={norm_val}"
                self._add_term(term)
                df[term] = len(set(titles))

        if self.cfg.include_plot_keywords:
            for kw, titles in dataset.PLOT_KEYWORD_INDEX.items():
                term = f"{self.cfg.plot_field}={kw}"
                self._add_term(term)
                df[term] = len(set(titles))

        # Smooth IDF: log((1+N)/(1+df)) + 1
        idf_arr = np.ones(len(self.idx2term), dtype=float)
        for i, term in enumerate(self.idx2term):
            dfi = df.get(term, 0)
            idf_arr[i] = math.log((1.0 + N) / (1.0 + dfi)) + 1.0
        self.idf = idf_arr

    def _build_embed_index(self):
        if not _SKLEARN_AVAILABLE:
            return None
        index = {}
        for field in self.cfg.topic_fields:
            if field not in dataset.ATTRIBUTE_CANONICAL:
                continue
            values = [v for v in dataset.ATTRIBUTE_CANONICAL[field].keys() if v]
            if not values:
                continue
            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                lowercase=True,
            )
            matrix = vectorizer.fit_transform(values)
            index[field] = {
                "vectorizer": vectorizer,
                "matrix": matrix,
                "candidates": values,
            }
        return index

    @property
    def vocab_size(self) -> int:
        return len(self.idx2term)

    def _extract_fielded_terms_heuristic(self, text: str) -> Dict[str, Set[str]]:
        """Return terms grouped by field using catalog grounding.

        Output uses *fielded concept tokens* of the form 'field=value', e.g.:
            {'actor': {'actor=dev patel'}, 'genre': {'genre=thriller'}, ...}

        Notes:
        - We use simple lexical grounding against catalog values (normalized substring match).
        - Plot keywords are treated as a controlled vocabulary and emitted as 'plot_kw=<kw>' tokens.
        """
        grouped: Dict[str, Set[str]] = {f: set() for f in self.cfg.topic_fields}
        lowered = (text or "").lower()

        # Structured fields (actor/director/genre/...)
        for field in self.cfg.topic_fields:
            if field not in dataset.ATTRIBUTE_CANONICAL:
                continue
            for norm_val in dataset.ATTRIBUTE_CANONICAL[field].keys():
                if norm_val and norm_val in lowered:
                    grouped.setdefault(field, set()).add(f"{field}={norm_val}")

        # Plot keyword grounding (controlled lexicon derived from catalog plots)
        if self.cfg.include_plot_keywords:
            for kw in dataset.find_plot_keywords_in_text(lowered):
                grouped.setdefault(self.cfg.plot_field, set()).add(f"{self.cfg.plot_field}={kw}")

        return grouped

    def _extract_fielded_terms_llm(self, text: str) -> Dict[str, Set[str]]:
        """Return terms grouped by field using an LLM extractor (optional)."""
        grouped: Dict[str, Set[str]] = {f: set() for f in self.cfg.topic_fields}
        if not text or not text.strip():
            return grouped
        llm_fields = [f for f in self.cfg.topic_fields if f != self.cfg.plot_field]
        prompt = (
            "Extract only these fields from the text: "
            + ", ".join(llm_fields)
            + ". Return STRICT JSON with keys for every field and values as lists of strings. "
            "If a field is absent, use an empty list. Do not include any extra text.\n\n"
            f"Text: {text}\n\nJSON:"
        )
        raw = _call_ollama(self.cfg.concept_extractor_model, prompt)
        payload = _extract_json(raw)
        try:
            data = json.loads(payload) if payload else {}
        except json.JSONDecodeError:
            data = {}
        for field in llm_fields:
            vals = data.get(field, [])
            if isinstance(vals, str):
                vals = [vals]
            if not isinstance(vals, list):
                continue
            for v in vals:
                norm = _canonicalize_value(field, v)
                if norm:
                    grouped[field].add(f"{field}={norm}")
        if self.cfg.include_plot_keywords:
            lowered = text.lower()
            for kw in dataset.find_plot_keywords_in_text(lowered):
                grouped.setdefault(self.cfg.plot_field, set()).add(f"{self.cfg.plot_field}={kw}")
        return grouped

    def _extract_fielded_terms_embed(self, text: str) -> Dict[str, Set[str]]:
        """Return terms grouped by field using grounded char-ngram similarity."""
        grouped: Dict[str, Set[str]] = {f: set() for f in self.cfg.topic_fields}
        if not text or not text.strip() or not self._embed_index:
            return grouped
        for field in self.cfg.topic_fields:
            info = self._embed_index.get(field)
            if not info:
                continue
            vec = info["vectorizer"].transform([text])
            sims = _sk_cosine_similarity(vec, info["matrix"]).ravel()
            if sims.size == 0:
                continue
            idxs = np.argsort(-sims)[: self._embed_top_k]
            for idx in idxs:
                if sims[idx] >= self._embed_threshold:
                    norm = info["candidates"][idx]
                    grouped[field].add(f"{field}={norm}")
        if self.cfg.include_plot_keywords:
            lowered = text.lower()
            for kw in dataset.find_plot_keywords_in_text(lowered):
                grouped.setdefault(self.cfg.plot_field, set()).add(f"{self.cfg.plot_field}={kw}")
        return grouped

    def extract_fielded_terms(self, text: str) -> Dict[str, Set[str]]:
        """Return terms grouped by field using configured extractor."""
        if self.cfg.concept_extractor_mode == "llm":
            llm_grouped = self._extract_fielded_terms_llm(text)
            if llm_grouped and any(llm_grouped.values()):
                return llm_grouped
        if self.cfg.concept_extractor_mode == "embed":
            if self._embed_index is not None:
                return self._extract_fielded_terms_embed(text)
        return self._extract_fielded_terms_heuristic(text)

    def topic_set(self, text: str, fields: Optional[Iterable[str]] = None) -> Set[str]:
        """Legacy API: flat set of grounded concept tokens used for set overlap (Jaccard).

        In the thesis/docs we refer to this as the **concept/constraint set** extracted from text.
        Each element is a token like 'actor=dev patel' or 'plot_kw=heist'.
        """
        grouped = self.extract_fielded_terms(text)
        if fields is None:
            fields = list(grouped.keys())
        out: Set[str] = set()
        for f in fields:
            out |= grouped.get(f, set())
        return out

    def concept_set(self, text: str, fields: Optional[Iterable[str]] = None) -> Set[str]:
        """Preferred API name (thesis terminology). Alias of :meth:`topic_set`."""
        return self.topic_set(text, fields=fields)

    def concepts_by_field(self, text: str, fields: Optional[Iterable[str]] = None) -> Dict[str, Set[str]]:
        """Return a mapping field -> concept tokens for diagnostics."""
        grouped = self.extract_fielded_terms(text)
        if fields is None:
            return grouped
        allowed = set(fields)
        return {f: grouped.get(f, set()) for f in allowed}

    def attribute_vector(
        self,
        text: str,
        active_fields: Optional[Iterable[str]] = None,
        binary: bool = True,
    ) -> np.ndarray:
        """IDF-weighted vector over fielded catalog terms.

        active_fields:
          If provided, restricts features to those fields (plus plot if enabled).
          This is useful to compute "within-topic" attribute similarity based on
          what the USER actually specified.
        """
        if self.idf is None:
            self._build_vocab_and_idf()
        vec = np.zeros(len(self.idx2term), dtype=float)

        grouped = self.extract_fielded_terms(text)
        allowed: Optional[Set[str]] = set(active_fields) if active_fields is not None else None
        plot_field = self.cfg.plot_field

        for field, terms in grouped.items():
            if allowed is not None and field not in allowed and field != plot_field:
                continue
            for term in terms:
                idx = self.term2idx.get(term)
                if idx is None:
                    continue
                vec[idx] = 1.0 if binary else vec[idx] + 1.0

        # Apply IDF
        vec *= self.idf
        return vec


# A module-level default featurizer (safe to reuse across evaluations)
DEFAULT_FEATURIZER = CatalogFeaturizer()

def extract_title_mentions(text: str) -> List[str]:
    """Heuristic title extraction for recommendation proxy (not used in TAS)."""
    lowered = (text or "").lower()
    matches: List[str] = []
    for norm_title, canonical in dataset.ATTRIBUTE_CANONICAL.get(TITLE_FIELD, {}).items():
        if norm_title and norm_title in lowered:
            matches.append(canonical)
    # Preserve order, remove duplicates
    return list(dict.fromkeys(matches))
