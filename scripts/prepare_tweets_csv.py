import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))

train_path = os.path.join(DATA_DIR, 'train.csv')
target_path = os.path.join(DATA_DIR, 'tweets.csv')

# Check if train.csv exists
if not os.path.exists(train_path):
    raise FileNotFoundError(f"train.csv not found at: {train_path}")

# Load Kaggle training dataset
train_df = pd.read_csv(train_path)

# Validate required columns
required_columns = {'text', 'target'}

if not required_columns.issubset(train_df.columns):
    raise ValueError("train.csv must contain 'text' and 'target' columns")

# Keep only required columns and rename them
tweets_df = train_df[['text', 'target']].rename(
    columns={
        'text': 'tweet',
        'target': 'label'
    }
)

print(f"Loaded {len(tweets_df)} rows from train.csv")

# Save processed dataset
tweets_df.to_csv(target_path, index=False)

print(f"Saved processed dataset to: {target_path}")
print(f"Final dataset size: {len(tweets_df)} rows")