"""
Evaluate model performance on test set.

Workflow:
1. Load saved model.pkl, vectorizer.pkl, X_test.pkl, y_test.pkl
2. Make predictions on test features
3. Calculate: accuracy, precision, recall, f1, confusion_matrix
4. Display formatted results and confusion matrix
"""

import pickle
import os
import sys
from xml.parsers.expat import model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
for path in (SRC_DIR, BASE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


def load_artifacts(model_path=None, vectorizer_path=None, X_test_path=None, y_test_path=None):
    """Load trained model, vectorizer, and test data."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
    if vectorizer_path is None:
        vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')
    if X_test_path is None:
        X_test_path = os.path.join(BASE_DIR, 'models', 'X_test.pkl')
    if y_test_path is None:
        y_test_path = os.path.join(BASE_DIR, 'models', 'y_test.pkl')

    required_files = [
        (model_path, 'Model'),
        (vectorizer_path, 'Vectorizer'),
        (X_test_path, 'Test features'),
        (y_test_path, 'Test labels')
    ]

    for path, name in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at {path}\nRun: python train.py")

    print("Loading trained artifacts...\n")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"  Model loaded from {model_path}")

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    print(f"  Vectorizer loaded from {vectorizer_path}")

    with open(X_test_path, 'rb') as f:
        tweets_test = pickle.load(f)
    print(f"  Test features loaded from {X_test_path}")

    with open(y_test_path, 'rb') as f:
        y_test = pickle.load(f)
    print(f"  Test labels loaded from {y_test_path}\n")

    return model, vectorizer, tweets_test, y_test


def evaluate_model(model, vectorizer, tweets_test, y_test):
    """Evaluate model: make predictions and compute metrics."""
    print("Evaluating model on test data...\n")

    X_test = vectorizer.transform(tweets_test)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def display_results(metrics):
    """Display evaluation metrics and confusion matrix."""

    cm = metrics['confusion_matrix']

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)

    print("\nPerformance Metrics")
    print("-" * 60)

    print(f"Accuracy : {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall   : {metrics['recall']:.2%}")
    print(f"F1-Score : {metrics['f1']:.2%}")

    print("\nConfusion Matrix")
    print("-" * 60)

    print(f"""
                Predicted
              No        Yes

Actual No   {cm[0,0]:5d}     {cm[0,1]:5d}
Actual Yes  {cm[1,0]:5d}     {cm[1,1]:5d}
""")

    print("Legend:")
    print(f"TN = {cm[0,0]} | FP = {cm[0,1]} | FN = {cm[1,0]} | TP = {cm[1,1]}")

    print("\nEvaluation complete!")
    print("=" * 60)


def main():
    """Main evaluation pipeline: load → evaluate → display results."""
    print("=" * 70)
    print("DISASTER TWEET CLASSIFIER - EVALUATION")
    print("=" * 70)
    print()

    try:
        model, vectorizer, tweets_test, y_test = load_artifacts()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    metrics = evaluate_model(
        model,
        vectorizer,
        tweets_test,
        y_test
    )
    display_results(metrics)


if __name__ == "__main__":
    main()
