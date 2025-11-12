# train_lda.py
"""
One-time script to train the LDA topic model on the movie dataset
and save the model and dictionary to disk for later use.
"""
import os
from gensim import corpora
from gensim.models import LdaModel

from dataset import MOVIE_DB
from evaluate import _preprocess_lda_text # Re-use the preprocessing logic
from config import LDA_MODEL_PATH, LDA_DICT_PATH, LDA_MODEL_DIR

def train_and_save_lda():
    """Trains and saves the LDA model."""
    os.makedirs(LDA_MODEL_DIR, exist_ok=True)

    print("--- Starting LDA model training ---")
    
    # 1. Load corpus
    plots = [movie.get('plot', '') for movie in MOVIE_DB.values() if movie.get('plot')]
    if not plots:
        print("Error: No movie plots found in dataset. Cannot train LDA model.")
        return

    print(f"Found {len(plots)} movie plots for training.")

    # 2. Preprocess documents
    processed_docs = [_preprocess_lda_text(plot) for plot in plots]
    
    # 3. Create dictionary and corpus
    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5) # Remove rare and common words
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    print(f"Dictionary created with {len(dictionary)} unique tokens.")

    # 4. Train LDA model
    print("Training LDA model...")
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=50, passes=15, random_state=42)

    # 5. Save the model and dictionary
    lda_model.save(LDA_MODEL_PATH)
    dictionary.save(LDA_DICT_PATH)
    print(f"--- LDA model and dictionary saved successfully to '{LDA_MODEL_DIR}' ---")

if __name__ == "__main__":
    train_and_save_lda()