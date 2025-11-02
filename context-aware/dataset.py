# dataset.py
# Tiny genre→movie mapping to simulate recommendations

def recommend(constraints):
    genre = constraints.get("genre", "")
    database = {
        "romance": [{"title": "The Notebook"}, {"title": "Pride and Prejudice"}, {"title": "La La Land"}],
        "horror": [{"title": "The Conjuring"}, {"title": "Get Out"}, {"title": "A Quiet Place"}],
        "sci-fi": [{"title": "Interstellar"}, {"title": "Blade Runner 2049"}, {"title": "The Matrix"}]
    }
    return database.get(genre, [{"title": "Inception"}])
