#!/usr/bin/env python3
"""
Train an LDA topic model on the movie plot corpus and save the model artifacts.
"""

import os

from gensim import corpora
from gensim.models import LdaModel

from dataset import MOVIE_DB
from evaluate import _preprocess_lda_text
from config import LDA_MODEL_PATH, LDA_DICT_PATH, LDA_MODEL_DIR


def train_and_save_lda():
    os.makedirs(LDA_MODEL_DIR, exist_ok=True)

    print("--- Starting LDA model training ---")
    plots = [movie.get("plot", "") for movie in MOVIE_DB.values() if movie.get("plot")]
    if not plots:
        print("Error: No movie plots found in dataset. Cannot train LDA model.")
        return

    print(f"Found {len(plots)} movie plots for training.")
    processed_docs = [_preprocess_lda_text(plot) for plot in plots]

    dictionary = corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    print(f"Dictionary created with {len(dictionary)} unique tokens.")
    print("Training LDA model...")
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=50, passes=15, random_state=42)

    lda_model.save(LDA_MODEL_PATH)
    dictionary.save(LDA_DICT_PATH)
    print(f"--- LDA model and dictionary saved successfully to '{LDA_MODEL_DIR}' ---")


if __name__ == "__main__":
    train_and_save_lda()
