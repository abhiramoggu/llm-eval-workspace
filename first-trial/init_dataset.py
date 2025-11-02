# dataset.py

# Fake dataset of movies with genres
FAKE_MOVIES = [
    {"title": "Titanic", "genre": "romance"},
    {"title": "The Ring", "genre": "horror"},
    {"title": "Inception", "genre": "sci-fi"},
    {"title": "La La Land", "genre": "romance"},
    {"title": "Saw", "genre": "horror"},
]

def recommend(constraints: dict):
    """
    Very simple recommender:
    - Input: constraints like {"genre": "romance"}
    - Output: a list of movies matching that genre
    """
    genre = constraints.get("genre")
    results = [m for m in FAKE_MOVIES if m["genre"] == genre]
    return results[:2]  # return top-2 matches for simplicity
