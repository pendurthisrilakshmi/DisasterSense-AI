"""
Train disaster classifier and save artifacts.

Workflow:
1. Load tweets.csv
2. Clean tweets using preprocess_tweet()
3. Split into 80% train, 20% test
4. Vectorize with TfidfVectorizer (max_features=5000)
5. Train LogisticRegression model
6. Save: model.pkl, vectorizer.pkl, X_test.pkl, y_test.pkl
"""

import pandas as pd
import pickle
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
for path in (SRC_DIR, BASE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from preprocess import preprocess_tweet


def load_dataset(csv_path):
    """Load tweets and labels from CSV."""
    print(f"Loading dataset from {csv_path}...")

    df = pd.read_csv(csv_path)

    tweets = df['tweet'].values
    labels = df['label'].values

    print(f"Loaded {len(tweets)} tweets")
    print(f"  Disaster tweets: {sum(labels)}")
    print(f"  Non-disaster tweets: {len(labels) - sum(labels)}\n")

    return tweets, labels


def preprocess_tweets(tweets):
    """Preprocess all tweets using preprocess_tweet()."""
    print("Preprocessing tweets...")
    processed_tweets = []

    for i, tweet in enumerate(tweets):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(tweets)}")

        cleaned_tweet = preprocess_tweet(tweet)
        processed_tweets.append(cleaned_tweet)

    print(f"Completed preprocessing {len(processed_tweets)} tweets\n")
    return processed_tweets


def vectorize_tweets(tweets_train, tweets_test, max_features=5000):
    """Convert tweets to TF-IDF vectors."""
    print("Vectorizing tweets with TF-IDF...")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )

    X_train_tfidf = vectorizer.fit_transform(tweets_train)
    X_test_tfidf = vectorizer.transform(tweets_test)

    print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"Training features shape: {X_train_tfidf.shape}")
    print(f"Test features shape: {X_test_tfidf.shape}\n")

    return vectorizer, X_train_tfidf, X_test_tfidf


def train_model(X_train, y_train):
    """Train logistic regression model."""
    print("Training Logistic Regression model...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("Model trained successfully\n")
    return model


def save_models(model, vectorizer, tweets_test, y_test, model_path=None, vectorizer_path=None, X_test_path=None, y_test_path=None):
    """Save model, vectorizer, and test data for later evaluation."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
    if vectorizer_path is None:
        vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
    if X_test_path is None:
        X_test_path = os.path.join(BASE_DIR, 'models', 'X_test.pkl')
    if y_test_path is None:
        y_test_path = os.path.join(BASE_DIR, 'models', 'y_test.pkl')

    print("Saving trained artifacts...")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print("  Model saved to ../models/model.pkl")

    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print("  Vectorizer saved to ../models/vectorizer.pkl")

    with open(X_test_path, 'wb') as f:
        pickle.dump(tweets_test, f)
    print("  Test tweets saved to ../models/X_test.pkl")

    with open(y_test_path, 'wb') as f:
        pickle.dump(y_test, f)
    print("  Test labels saved to ../models/y_test.pkl")


def main():
    """Main training pipeline: load → preprocess → split → vectorize → train → save."""
    print("=" * 60)
    print("DISASTER TWEET CLASSIFIER - TRAINING PIPELINE")
    print("=" * 60 + "\n")

    csv_path = os.path.join(BASE_DIR, 'data', 'tweets.csv')

    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found!")
        print("Please ensure data/tweets.csv exists with 'tweet' and 'label' columns.")
        return

    tweets, labels = load_dataset(csv_path)
    processed_tweets = preprocess_tweets(tweets)

    print("Splitting data: 80% train, 20% test")
    tweets_train, tweets_test, y_train, y_test = train_test_split(
    processed_tweets,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)
    print(f"  Train samples: {len(tweets_train)}")
    print(f"  Test samples: {len(tweets_test)}\n")

    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_tweets(tweets_train, tweets_test)

    model = train_model(X_train_tfidf, y_train)

    save_models(model, vectorizer, tweets_test, y_test)

    print("=" * 60)
    print("Training complete! Ready for evaluation.")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
