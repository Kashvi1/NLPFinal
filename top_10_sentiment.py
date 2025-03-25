import pandas as pd

def get_top_bottom_sentiments(csv_file, output_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure the required columns exist
    if 'business_id' not in df.columns or 'avg_sentiment_score' not in df.columns:
        raise ValueError("CSV file must contain 'business_id' and 'avg_sentiment_score' columns")
    
    # Sort by sentiment_score
    df_sorted = df.sort_values(by='avg_sentiment_score', ascending=False)
    
    # Get top 10 and bottom 10 sentiment scores
    top_10 = df_sorted.head(10)
    bottom_10 = df_sorted.tail(10)
    
    # Save to a text file
    with open(output_file, 'w') as file:
        file.write("Top 10 Sentiments:\n")
        file.write(top_10.to_string(index=False))
        file.write("\n\nBottom 10 Sentiments:\n")
        file.write(bottom_10.to_string(index=False))
    
    print(f"Top 10 and Bottom 10 sentiments saved to {output_file}")

# Example usage
csv_file = 'review_sentiment.csv'  # Replace with your actual file path
output_file = 'sentiments_output.txt'  # Desired output file path
get_top_bottom_sentiments(csv_file, output_file)
