"""
Make predictions on new tweets.

Workflow:
1. Load model.pkl and vectorizer.pkl
2. Accept tweet input (interactive or demo mode)
3. Clean tweet using preprocess_tweet()
4. Vectorize with loaded vectorizer
5. Get prediction (0 or 1) and confidence probability
6. Display classification result

Usage: python predict.py           # Interactive mode
       python predict.py demo      # Demo with examples
"""

import pickle
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
for path in (SRC_DIR, BASE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from preprocess import preprocess_tweet


def load_models(model_path=None, vectorizer_path=None):
    """Load trained model and vectorizer."""
    if model_path is None:
        model_path = os.path.join(BASE_DIR, 'models', 'model.pkl')
    if vectorizer_path is None:
        vectorizer_path = os.path.join(BASE_DIR, 'models', 'vectorizer.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer not found at {vectorizer_path}")

    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer


def predict_tweet(tweet, model, vectorizer):
    """Preprocess, vectorize, and get prediction with confidence."""
    cleaned_tweet = preprocess_tweet(tweet)
    tweet_vector = vectorizer.transform([cleaned_tweet])

    prediction = model.predict(tweet_vector)[0]
    probabilities = model.predict_proba(tweet_vector)[0]

    prob_non_disaster = probabilities[0]
    prob_disaster = probabilities[1]

    if prediction == 0:
        classification = "Non-Disaster"
        confidence = prob_non_disaster
    else:
        classification = "Disaster"
        confidence = prob_disaster

    return {
        'classification': classification,
        'prediction': prediction,
        'confidence': confidence,
        'prob_non_disaster': prob_non_disaster,
        'prob_disaster': prob_disaster,
        'cleaned_tweet': cleaned_tweet
    }


def display_prediction(tweet, result):
    """Display prediction result."""
    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print(f"\nOriginal Tweet:\n  {tweet}")
    print("Cleaned Tweet:")

    if result['cleaned_tweet'].strip():
        print(f"  {result['cleaned_tweet']}")
    else:
        print("  [No meaningful words remaining after preprocessing]")
    print(f"\n{'Classification:':<20} {result['classification']}")
    print(f"{'Confidence:':<20} {result['confidence']:.2%}")
    print(f"\nProbability Breakdown:")
    print(f"  Non-Disaster: {result['prob_non_disaster']:.2%}")
    print(f"  Disaster:     {result['prob_disaster']:.2%}")
    print("=" * 70 + "\n")


def interactive_prediction():
    """Accept user tweets and display predictions."""
    print("\n" + "=" * 70)
    print("DISASTER TWEET CLASSIFIER - INTERACTIVE PREDICTION")
    print("=" * 70)
    print("\nType 'quit', 'exit', or 'q' to stop\n")

    try:
        model, vectorizer = load_models()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first: python train.py")
        return

    while True:
        tweet = input("Enter a tweet: ").strip()

        if tweet.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not tweet:
            print("Please enter a tweet\n")
            continue

        try:
            result = predict_tweet(tweet, model, vectorizer)
            display_prediction(tweet, result)
        except Exception as e:
            print(f"Error making prediction: {e}\n")


def demo():
    """Show predictions on example tweets."""
    print("\n" + "=" * 70)
    print("DISASTER TWEET CLASSIFIER - DEMO")
    print("=" * 70)

    example_tweets = [
        "There is a massive fire downtown!",
        "Just had my morning coffee, wonderful day!",
        "EARTHQUAKE!! Stay safe everyone!",
        "Watching the sunset at the beach",
        "Flood warning issued for the area",
        "Can't wait for the weekend",
        "Hurricane approaching the coast",
    ]

    try:
        model, vectorizer = load_models()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first: python train.py")
        return

    print("\nMaking predictions on example tweets...\n")

    for tweet in example_tweets:
        result = predict_tweet(tweet, model, vectorizer)

        emoji = "[DISASTER]" if result['prediction'] == 1 else "[SAFE]"
        print(f"[{emoji}] {tweet}")
        print(f"   -> {result['classification']} ({result['confidence']:.1%} confidence)\n")


def main():
    """Interactive mode by default, demo mode with 'demo' argument."""
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'demo':
        demo()
    else:
        interactive_prediction()


if __name__ == "__main__":
    main()

