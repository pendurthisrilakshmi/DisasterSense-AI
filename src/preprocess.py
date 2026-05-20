"""
PREPROCESSING MODULE
=====================

This module handles all text preprocessing for the disaster tweet classifier.

Why Preprocessing?
------------------
Raw tweets contain:
- URLs
- Punctuation
- Extra whitespace
- Mixed case
- Stopwords (common words like 'the', 'a', 'is')

These make it harder for the model to find patterns. Preprocessing cleans the text
so the model can focus on important words.

Preprocessing Steps:
1. Lowercase: "FIRE" → "fire" (consistent)
2. Remove URLs: "Check http://example.com" → "Check"
3. Remove punctuation: "Hello!" → "Hello"
4. Remove stopwords: "the fire is bad" → "fire bad"
5. Tokenize: Split into individual words for vectorization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

# Download required NLTK datasets (only need once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load English stopwords
# Stopwords are common words that don't add meaning: 'the', 'a', 'is', 'and'
STOP_WORDS = set(stopwords.words('english'))


def remove_urls(text):
    """
    Remove URLs from text.
    
    Example:
        Input: "Check http://example.com for info"
        Output: "Check for info"
    
    Args:
        text (str): Raw tweet text
        
    Returns:
        str: Text without URLs
    """
    # Regex pattern to match URLs starting with http:// or https://
    url_pattern = url_pattern = r'https?://\S+'
    return re.sub(url_pattern, '', text)


def remove_punctuation(text):
    """
    Remove punctuation and special characters.
    
    Example:
        Input: "Hello! How are you?"
        Output: "Hello How are you"
    
    Args:
        text (str): Text with punctuation
        
    Returns:
        str: Text without punctuation
    """
    # Keep only letters, numbers, and spaces
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)


def remove_stopwords(tokens):
    """
    Remove common stopwords from token list.
    
    Why remove stopwords?
    - "the", "a", "is" appear in all tweets
    - They don't help distinguish disaster vs non-disaster tweets
    - Removing them makes the model focus on meaningful words
    
    Example:
        Input: ['the', 'fire', 'is', 'near', 'my', 'home']
        Output: ['fire', 'near', 'home']
    
    Args:
        tokens (list): List of words
        
    Returns:
        list: Words without stopwords
    """
    return [word for word in tokens if word.lower() not in STOP_WORDS]


def preprocess_tweet(tweet):
    """
    Complete preprocessing pipeline for a single tweet.
    
    This function applies all preprocessing steps in order:
    1. Remove URLs
    2. Lowercase conversion
    3. Remove punctuation
    4. Tokenize (split into words)
    5. Remove stopwords
    
    Example:
        Input: "HELP! There's a fire http://example.com #disaster"
        Output: "help there fire disaster"
    
    Args:
        tweet (str): Raw tweet text
        
    Returns:
        str: Cleaned and preprocessed tweet
    """
    
    # Step 1: Remove URLs
    tweet = remove_urls(tweet)
    
    # Step 2: Convert to lowercase (unify case)
    tweet = tweet.lower()
    
    # Step 3: Remove punctuation
    tweet = remove_punctuation(tweet)
    
    # Step 4: Tokenize (split into individual words)
    tokenizer = RegexpTokenizer(r"\w+")
    tokens = tokenizer.tokenize(tweet)
    
    # Step 5: Remove stopwords
    tokens = remove_stopwords(tokens)
    
    # Join tokens back into a string
    cleaned_tweet = ' '.join(tokens)
    
    return cleaned_tweet


if __name__ == "__main__":
    # Test the preprocessing pipeline
    # print("=== PREPROCESSING TEST ===\n")
    
    sample_tweets = [
        "HELP! There's a fire near my house! Check http://example.com #disaster",
        "Just finished my coffee, it's a beautiful day",
        "EARTHQUAKE!!! Stay safe everyone! http://twitter.com #emergency"
    ]
    
    for tweet in sample_tweets:
        cleaned = preprocess_tweet(tweet)
        print(f"Original: {tweet}")
        print(f"Cleaned:  {cleaned}")
        print()
