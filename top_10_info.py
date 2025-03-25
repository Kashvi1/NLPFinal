import pandas as pd

# Load the business information CSV
business_info_df = pd.read_csv('cleaned_yelp_business_data (2).csv')

# Read the entire .txt file into a DataFrame, skipping the first few lines
# This will handle both Top and Bottom 10 Sentiments sections
with open('path_to_top_10_sentiments.txt', 'r') as file:
    lines = file.readlines()

# Find the indices where "Top 10 Sentiments" and "Bottom 10 Sentiments" sections start
top_start_idx = lines.index('Top 10 Sentiments:\n') + 2  # Skip the header line
bottom_start_idx = lines.index('Bottom 10 Sentiments:\n') + 2  # Skip the header line

# Extract the top 10 sentiment data
top_10_data = lines[top_start_idx:bottom_start_idx-2]

# Convert to a DataFrame
top_10_sentiments_df = pd.DataFrame([line.strip().split() for line in top_10_data], columns=['business_id', 'avg_sentiment_score'])

# Extract business IDs of the top 10 businesses
top_10_business_ids = top_10_sentiments_df['business_id'].tolist()

# Filter the business information DataFrame to include only the top 10 businesses
top_10_business_info = business_info_df[business_info_df['business_id'].isin(top_10_business_ids)]

# Save the filtered business information to a CSV file
top_10_business_info.to_csv('top_10_business_info.csv', index=False)

# Optional: Display the filtered business information
print(top_10_business_info)
