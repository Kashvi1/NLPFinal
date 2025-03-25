import pandas as pd
from textblob import TextBlob
from tqdm import tqdm  # Progress bar library

# ==== CONFIGURATION ====
INPUT_CSV_PATH = "/Users/kashvi/Downloads/cleaned_data/cleaned_yelp_review_data.csv"  # Update with actual file path
OUTPUT_CSV_PATH = "/Users/kashvi/Downloads/NLPFinal/review_sentiment.csv"  # Update with desired output file path
CHUNK_SIZE = 100000  # Adjust for memory constraints

# Function to compute sentiment polarity
def get_sentiment(text):
    """Returns sentiment polarity score from -1 (negative) to +1 (positive)."""
    return TextBlob(text).sentiment.polarity if isinstance(text, str) else 0

# Dictionary to store sentiment scores per business
business_sentiments = {}

# Get total rows for progress bar (optional: speeds up if file is large)
total_rows = sum(1 for _ in open(INPUT_CSV_PATH)) - 1  # Subtract header row

# Process CSV in chunks with progress bar
with tqdm(total=total_rows, desc="Processing Reviews", unit=" rows") as pbar:
    for chunk in pd.read_csv(INPUT_CSV_PATH, usecols=["business_id", "text"], chunksize=CHUNK_SIZE):
        chunk.dropna(subset=["text"], inplace=True)  # Remove rows with missing text
        chunk["sentiment_score"] = chunk["text"].apply(get_sentiment)

        # Aggregate sentiment scores per business ID
        for business_id, score in zip(chunk["business_id"], chunk["sentiment_score"]):
            if business_id in business_sentiments:
                business_sentiments[business_id].append(score)
            else:
                business_sentiments[business_id] = [score]

        # Update progress bar
        pbar.update(len(chunk))

# Compute the average sentiment per business
business_avg_sentiment = {
    bid: sum(scores) / len(scores) for bid, scores in business_sentiments.items()
}

# Save results to CSV
sentiment_df = pd.DataFrame(list(business_avg_sentiment.items()), columns=["business_id", "avg_sentiment_score"])
sentiment_df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"Sentiment analysis completed! Results saved to: {OUTPUT_CSV_PATH}")
