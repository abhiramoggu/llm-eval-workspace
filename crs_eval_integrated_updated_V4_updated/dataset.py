# dataset.py
# Movie metadata utilities for the simulator / CRS
import json
import os
import random
import re
from collections import defaultdict
from typing import Dict, List, Iterable, Set

MOVIE_DB: Dict[str, dict] = {}
GENRE_MAP: Dict[str, List[str]] = defaultdict(list)

# Attribute indexes -> movie title lists
ATTRIBUTE_FIELDS = ("name", "genre", "actor", "director", "language", "writer", "year")
ATTRIBUTE_INDEX: Dict[str, Dict[str, List[str]]] = {field: defaultdict(list) for field in ATTRIBUTE_FIELDS}
ATTRIBUTE_VALUES: Dict[str, Set[str]] = {field: set() for field in ATTRIBUTE_FIELDS}
ATTRIBUTE_CANONICAL: Dict[str, Dict[str, str]] = {field: {} for field in ATTRIBUTE_FIELDS}

PLOT_KEYWORD_INDEX: Dict[str, List[str]] = defaultdict(list)
PLOT_KEYWORDS: Set[str] = set()
PLOT_STOPWORDS = {
    "the", "and", "with", "that", "this", "from", "into", "about", "their", "they",
    "them", "have", "when", "where", "after", "before", "toward", "towards", "against",
    "while", "over", "under", "through", "also", "film", "movie", "story", "stories",
    "life", "lives", "takes", "place", "city", "town", "family", "people"
}

CONSTRAINT_ALIASES = {
    "title": "name",
    "plot_keywords": "plot",
}
SUPPORTED_CONSTRAINTS = set(ATTRIBUTE_FIELDS) | {"plot"}


def _normalize_genre_name(name: str) -> str:
    """Simplifies genre names, e.g., 'Romance Film' -> 'romance'."""
    if not name:
        return ""
    return (
        name.lower()
        .replace(" film", "")
        .replace(" movie", "")
        .replace(" fiction", "")
        .strip()
    )


def _normalize_value(field: str, value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if field == "genre":
        return _normalize_genre_name(text)
    if field == "year":
        return re.sub(r"[^0-9]", "", text)
    return text.lower()


def _extract_plot_keywords(plot: str) -> List[str]:
    if not plot:
        return []
    tokens = re.findall(r"[a-zA-Z]{4,}", plot.lower())
    keywords = [tok for tok in tokens if tok not in PLOT_STOPWORDS]
    return list(dict.fromkeys(keywords))


def _add_to_index(field: str, value: str, title: str):
    norm = _normalize_value(field, value)
    if not norm:
        return
    ATTRIBUTE_INDEX[field][norm].append(title)
    ATTRIBUTE_VALUES[field].add(str(value).strip())
    ATTRIBUTE_CANONICAL[field].setdefault(norm, str(value).strip())


def _index_movie(title: str, details: dict):
    safe_details = details or {}
    safe_details.setdefault("name", title)
    for field in ATTRIBUTE_FIELDS:
        values: Iterable[str]
        if field == "name":
            values = [safe_details.get("name", title)]
        elif field == "year":
            values = [safe_details.get("year")]
        else:
            values = safe_details.get(field, [])
        for value in values or []:
            _add_to_index(field, value, title)
            if field == "genre":
                GENRE_MAP[_normalize_genre_name(value)].append(title)

    for keyword in _extract_plot_keywords(safe_details.get("plot", "")):
        PLOT_KEYWORDS.add(keyword)
        PLOT_KEYWORD_INDEX[keyword].append(title)


def _load_movie_data():
    """Loads and processes the movie data from the JSON file."""
    global MOVIE_DB
    script_dir = os.path.dirname(__file__)
    json_path = os.path.join(script_dir, "opendialkg_movie_data.json")

    with open(json_path, "r") as f:
        data = json.load(f)

    MOVIE_DB = data

    for title, details in data.items():
        _index_movie(title, details)


# Load the data when the module is first imported
_load_movie_data()


def get_available_genres(min_size: int = 3) -> List[str]:
    """Returns normalized genres that have sufficient coverage."""
    return [
        genre
        for genre, titles in GENRE_MAP.items()
        if len(titles) >= min_size and len(genre) > 2
    ]


def get_attribute_values(attribute: str, min_count: int = 1) -> List[str]:
    """Expose possible values for user simulation / heuristics."""
    attr = CONSTRAINT_ALIASES.get(attribute, attribute)
    if attr not in ATTRIBUTE_INDEX:
        if attr == "plot":
            return sorted(list(PLOT_KEYWORDS))
        return []
    values = [
        ATTRIBUTE_CANONICAL[attr][norm]
        for norm, titles in ATTRIBUTE_INDEX[attr].items()
        if len(titles) >= min_count
    ]
    return sorted(values)


def find_attribute_in_text(attribute: str, text: str) -> str | None:
    """Return the first attribute value that appears in the given text."""
    attr = CONSTRAINT_ALIASES.get(attribute, attribute)
    if attr not in ATTRIBUTE_INDEX:
        return None
    lowered = text.lower()
    for norm_value, canonical in ATTRIBUTE_CANONICAL[attr].items():
        if norm_value and norm_value in lowered:
            return canonical
    return None


def find_plot_keywords_in_text(text: str, limit: int = 3) -> List[str]:
    """Extract plot keywords that overlap with indexed plots."""
    lowered_tokens = _extract_plot_keywords(text)
    matches = []
    for token in lowered_tokens:
        if token in PLOT_KEYWORD_INDEX:
            matches.append(token)
        if len(matches) >= limit:
            break
    return matches


def _normalize_constraints(constraints: Dict[str, object] | None) -> Dict[str, List[str]]:
    normalized: Dict[str, List[str]] = {}
    if not constraints:
        return normalized
    for raw_key, raw_val in constraints.items():
        if raw_val in (None, "", []):
            continue
        key = CONSTRAINT_ALIASES.get(raw_key, raw_key)
        if key not in SUPPORTED_CONSTRAINTS:
            continue
        values: List[str]
        if isinstance(raw_val, (list, tuple, set)):
            values = [str(v).strip() for v in raw_val if v]
        else:
            values = [str(raw_val).strip()]
        values = [v for v in values if v]
        if not values:
            continue
        normalized.setdefault(key, []).extend(values)
    return normalized


def _lookup_titles(field: str, value: str) -> List[str]:
    if field == "plot":
        tokens = _extract_plot_keywords(value)
        titles: List[str] = []
        for token in tokens:
            titles.extend(PLOT_KEYWORD_INDEX.get(token, []))
        return titles

    if field not in ATTRIBUTE_INDEX:
        return []

    norm = _normalize_value(field, value)
    direct = ATTRIBUTE_INDEX[field].get(norm, [])
    if direct:
        return direct

    # Fallback: substring match for partial mentions
    matches: List[str] = []
    for candidate_norm, titles in ATTRIBUTE_INDEX[field].items():
        if norm and norm in candidate_norm:
            matches.extend(titles)
    return matches


def recommend(constraints: Dict[str, object] | None = None, limit: int = 30) -> List[dict]:
    """Recommend movies based on mixed constraints (genre, actor, year, etc.)."""
    normalized = _normalize_constraints(constraints)
    candidate_sets: List[Set[str]] = []

    for field, values in normalized.items():
        matched_titles: Set[str] = set()
        for value in values:
            for title in _lookup_titles(field, value):
                matched_titles.add(title)
        if matched_titles:
            candidate_sets.append(matched_titles)

    if candidate_sets:
        candidates = set.intersection(*candidate_sets) if len(candidate_sets) > 1 else candidate_sets[0]
        if not candidates:
            candidates = set.union(*candidate_sets)
    else:
        # default to all titles, keeping genre heuristics
        preferred_genre = normalized.get("genre", [])
        if preferred_genre:
            genre = _normalize_genre_name(preferred_genre[0])
            candidates = set(GENRE_MAP.get(genre, []))
        else:
            candidates = set(MOVIE_DB.keys())

    if not candidates:
        print(f"[Warning] No movies found for constraints: {constraints}")
        fallback = MOVIE_DB.get("Inception") or {"name": "Inception"}
        return [fallback]

    pool = [MOVIE_DB[title] for title in candidates if title in MOVIE_DB]
    random.shuffle(pool)
    limit = max(1, min(limit, 50))
    return pool[: min(len(pool), limit)]
