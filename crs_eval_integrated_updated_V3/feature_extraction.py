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
# from plot summaries (dataset.PLOT_KEYWORDS).
#
# NOTE: This module intentionally avoids any "open" text embedding that could
# drift away from the catalog grounding, because the thesis aims to show a
# grounded alternative to LLM-as-judge scoring.

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

import dataset
from config import CONCEPT_FIELDS


_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


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
    include_plot_keywords: bool = True


class CatalogFeaturizer:
    """Builds catalog-grounded topic sets and attribute vectors."""

    def __init__(self, cfg: Optional[FeaturizerConfig] = None):
        self.cfg = cfg or FeaturizerConfig()

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
                term = f"plot={kw}"
                self._add_term(term)
                df[term] = len(set(titles))

        # Smooth IDF: log((1+N)/(1+df)) + 1
        idf_arr = np.ones(len(self.idx2term), dtype=float)
        for i, term in enumerate(self.idx2term):
            dfi = df.get(term, 0)
            idf_arr[i] = math.log((1.0 + N) / (1.0 + dfi)) + 1.0
        self.idf = idf_arr

    @property
    def vocab_size(self) -> int:
        return len(self.idx2term)

    def extract_fielded_terms(self, text: str) -> Dict[str, Set[str]]:
        """Return terms grouped by field.

        Output uses *fielded concept tokens* of the form 'field=value', e.g.:
            {'actor': {'actor=dev patel'}, 'genre': {'genre=thriller'}, ...}

        Notes:
        - We use simple lexical grounding against catalog values (normalized substring match).
        - Plot keywords are treated as a controlled vocabulary and emitted as 'plot=<kw>' tokens.
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
                grouped.setdefault("plot", set()).add(f"plot={kw}")

        return grouped

    def topic_set(self, text: str, fields: Optional[Iterable[str]] = None) -> Set[str]:
        """Legacy API: flat set of grounded concept tokens used for set overlap (Jaccard).

        In the thesis/docs we refer to this as the **concept/constraint set** extracted from text.
        Each element is a token like 'actor=dev patel' or 'plot=heist'.
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

        for field, terms in grouped.items():
            if allowed is not None and field not in allowed and field != "plot":
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
