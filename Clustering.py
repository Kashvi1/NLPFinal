import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
import gc
import sys
import os
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

# Display configuration information
print("="*50)
print("YELP REVIEW CLUSTERING FOR EVENT PLANNING")
print("="*50)
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*50)

# Function to report memory usage
def memory_usage(df):
    return f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"

# Create output directory for all results
output_dir = "clustering_results"
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {output_dir}")

try:
    # ====================== DATA LOADING ======================
    # Enter the FULL paths to your data files here
    business_file_path = '/Users/omniaabouhassan/Desktop/NLPFinal/cleaned_yelp_business_data (2).csv'
    review_file_path = '/Users/omniaabouhassan/Desktop/NLPFinal/cleaned_yelp_review_data.csv'
    
    # Check if files exist
    print(f"Checking if business file exists: {os.path.exists(business_file_path)}")
    print(f"Checking if review file exists: {os.path.exists(review_file_path)}")
    
    # Load business data with optimized dtypes
    print("\n1. Loading business data...")
    business_df = pd.read_csv(business_file_path, 
                             dtype={'business_id': 'str'}, 
                             usecols=['business_id', 'categories'])
    print(f"   Business data loaded: {len(business_df)} rows, Memory: {memory_usage(business_df)}")
    
    # Load review data with optimized dtypes
    print("\n2. Loading review data...")
    review_df = pd.read_csv(review_file_path, 
                           dtype={'business_id': 'str', 'stars': 'int8'}, 
                           usecols=['business_id', 'stars', 'text', 'date'])
    print(f"   Review data loaded: {len(review_df)} rows, Memory: {memory_usage(review_df)}")
    
    # ====================== DATA SAMPLING ======================
    # Take 15% sample of reviews (increased from 10% to get more data)
    print("\n3. Sampling review data...")
    sample_size = int(len(review_df) * 0.15)
    print(f"   Sample size: {sample_size} reviews (15% of total)")
    review_sample = review_df.sample(n=sample_size, random_state=42)
    del review_df  # Free memory
    gc.collect()
    print(f"   Sampled review data: {len(review_sample)} rows")
    
    # ====================== DATA MERGING ======================
    print("\n4. Merging business and review data...")
    merged_df = pd.merge(review_sample, business_df, on='business_id', how='inner')
    del review_sample, business_df  # Free memory
    gc.collect()
    print(f"   Merged data: {len(merged_df)} rows, Memory: {memory_usage(merged_df)}")
    
    # ====================== DATA PREPROCESSING ======================
    print("\n5. Preprocessing data...")
    # Remove missing values
    merged_df.dropna(subset=['text', 'stars', 'categories'], inplace=True)
    
    # Convert date to datetime
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    
    # Extract time-based features
    merged_df['year'] = merged_df['date'].dt.year
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['day_of_week'] = merged_df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # Calculate review recency (how recent is the review)
    most_recent_date = merged_df['date'].max()
    merged_df['days_since'] = (most_recent_date - merged_df['date']).dt.days
    merged_df['recency_score'] = 1 / (1 + np.log1p(merged_df['days_since']))  # Normalize to 0-1
    
    # Create sentiment labels and values
    print("   Creating sentiment features...")
    merged_df['sentiment_value'] = np.select(
        [merged_df['stars'] <= 2, merged_df['stars'] == 3, merged_df['stars'] > 3],
        [-1, 0, 1]
    )
    merged_df['sentiment'] = np.select(
        [merged_df['stars'] <= 2, merged_df['stars'] == 3, merged_df['stars'] > 3],
        ['negative', 'neutral', 'positive']
    )
    
    # Extract primary category
    merged_df['category_primary'] = merged_df['categories'].str.split(',').str[0]
    
    # Get top 10 categories
    top_categories = merged_df['category_primary'].value_counts().head(10).index.tolist()
    print(f"   Top 10 categories: {top_categories}")
    
    # Filter to top categories only
    top_df = merged_df[merged_df['category_primary'].isin(top_categories)].copy()
    del merged_df  # Free memory
    gc.collect()
    print(f"   Filtered to top categories: {len(top_df)} rows")
    
    # ====================== FEATURE ENGINEERING ======================
    print("\n6. Extracting text features...")
    # Calculate review length in small batches
    top_df['review_length'] = 0  # Initialize
    batch_size = 1000
    for i in range(0, len(top_df), batch_size):
        end_idx = min(i + batch_size, len(top_df))
        top_df.loc[top_df.index[i:end_idx], 'review_length'] = top_df['text'].iloc[i:end_idx].str.len()
        if i % 20000 == 0 and i > 0:
            print(f"   Processed {i} reviews...")
    
    # Create text complexity score (ratio of length to word count)
    top_df['word_count'] = top_df['text'].str.split().str.len()
    top_df['avg_word_length'] = top_df['review_length'] / top_df['word_count'].clip(lower=1)
    
    # Extract exclamation marks as a feature of emphasis/emotion
    top_df['exclamation_count'] = top_df['text'].str.count('!')
    
    # Create categorical encodings
    print("   Encoding categorical features...")
    category_dummies = pd.get_dummies(top_df['category_primary'], prefix='cat')
    day_dummies = pd.get_dummies(top_df['day_of_week'], prefix='day')
    
    # ====================== FEATURE SELECTION FOR CLUSTERING ======================
    print("\n7. Preparing features for clustering...")
    # Select numerical features for clustering
    base_features = [
        'stars',                  # Rating given by reviewer
        'sentiment_value',        # Derived sentiment (-1, 0, 1)
        'review_length',          # Length of review in characters
        'recency_score',          # How recent the review is (0-1)
        'avg_word_length',        # Text complexity measure
        'exclamation_count'       # Emotional emphasis measure
    ]
    
    # Prepare the feature matrix
    X = top_df[base_features].copy()
    
    # Handle any NaN values
    X.fillna(X.mean(), inplace=True)
    print(f"   Clustering data shape: {X.shape}")
    print(f"   Features used: {', '.join(base_features)}")
    
    # ====================== DATA STANDARDIZATION ======================
    print("\n8. Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_means = pd.Series(scaler.mean_, index=base_features)
    feature_stds = pd.Series(scaler.scale_, index=base_features)
    
    # ====================== OPTIMAL CLUSTER DETERMINATION ======================
    print("\n9. Finding optimal number of clusters...")
    # Calculate various metrics for different k values
    k_range = range(2, 8)
    inertia = []
    silhouette_scores = []
    
    for k in k_range:
        print(f"   Testing k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    # Plot elbow curve and silhouette scores
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal k')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, 'ro-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method for Optimal k')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/optimal_cluster_analysis.png")
    
    # ====================== KMEANS CLUSTERING ======================
    # Choose the best k based on the elbow and silhouette plots
    # Let's use 4 clusters for more nuanced grouping
    best_k = 4
    print(f"\n10. Performing K-means clustering with k={best_k}...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to dataframe
    top_df['cluster'] = cluster_labels
    
    # ====================== CLUSTER ANALYSIS ======================
    print("\n11. Analyzing cluster characteristics...")
    # Calculate cluster profiles
    cluster_profile = top_df.groupby('cluster')[base_features].mean()
    print("   Cluster profiles (mean values):")
    print(cluster_profile)
    
    # Calculate cluster sizes
    cluster_sizes = top_df['cluster'].value_counts().sort_index()
    print("\n   Cluster sizes:")
    for cluster, size in cluster_sizes.items():
        print(f"   Cluster {cluster}: {size} reviews ({size/len(top_df)*100:.1f}%)")
    
    # ====================== PCA VISUALIZATION ======================
    print("\n12. Generating PCA visualization...")
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a dataframe for visualization
    viz_df = pd.DataFrame({
        'PCA1': X_pca[:, 0],
        'PCA2': X_pca[:, 1],
        'Cluster': cluster_labels,
        'Sentiment': top_df['sentiment'],
        'Category': top_df['category_primary'],
        'Stars': top_df['stars']
    })
    
    # Plot clusters in PCA space
    plt.figure(figsize=(12, 10))
    
    # Main cluster plot
    plt.subplot(2, 2, 1)
    sns.scatterplot(
        data=viz_df.sample(min(10000, len(viz_df))),  # Sample for better visualization
        x='PCA1', 
        y='PCA2', 
        hue='Cluster',
        palette='viridis', 
        alpha=0.6
    )
    plt.title('Clusters in PCA Space')
    
    # By sentiment
    plt.subplot(2, 2, 2)
    sns.scatterplot(
        data=viz_df.sample(min(10000, len(viz_df))),
        x='PCA1', 
        y='PCA2', 
        hue='Sentiment',
        palette='RdYlGn', 
        alpha=0.6
    )
    plt.title('Sentiment in PCA Space')
    
    # By star rating
    plt.subplot(2, 2, 3)
    sns.scatterplot(
        data=viz_df.sample(min(10000, len(viz_df))),
        x='PCA1', 
        y='PCA2', 
        hue='Stars',
        palette='YlOrRd', 
        alpha=0.6
    )
    plt.title('Star Ratings in PCA Space')
    
    # By category (top 3 categories)
    top3_categories = top_df['category_primary'].value_counts().head(3).index
    plt.subplot(2, 2, 4)
    sns.scatterplot(
        data=viz_df[viz_df['Category'].isin(top3_categories)].sample(min(10000, len(viz_df))),
        x='PCA1', 
        y='PCA2', 
        hue='Category',
        palette='Set1', 
        alpha=0.6
    )
    plt.title('Top 3 Categories in PCA Space')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_visualization.png")
    
    # ====================== FEATURE IMPORTANCE ======================
    print("\n13. Analyzing feature importance...")
    # Create feature importance based on distance from cluster centers
    feature_importance = pd.DataFrame(
        np.abs(kmeans.cluster_centers_),
        columns=base_features
    )
    
    # Plot feature importance heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        feature_importance,
        cmap='YlGnBu',
        annot=True,
        fmt=".2f",
        xticklabels=base_features,
        yticklabels=[f"Cluster {i}" for i in range(best_k)]
    )
    plt.title('Feature Importance by Cluster')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    
    # ====================== SENTIMENT ANALYSIS BY CLUSTER ======================
    print("\n14. Analyzing sentiment distribution by cluster...")
    # Create sentiment distribution table
    sentiment_by_cluster = pd.crosstab(
        top_df['cluster'], 
        top_df['sentiment'], 
        normalize='index'
    )
    
    # Plot sentiment distribution
    plt.figure(figsize=(12, 6))
    sentiment_by_cluster.plot(
        kind='bar',
        stacked=True,
        colormap='RdYlGn',
        figsize=(12, 6)
    )
    plt.title('Sentiment Distribution by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.legend(title='Sentiment')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sentiment_by_cluster.png")
    
    # ====================== BUSINESS CATEGORY ANALYSIS ======================
    print("\n15. Analyzing business category distribution by cluster...")
    # Create category distribution table (top 6 categories for readability)
    top6_categories = top_df['category_primary'].value_counts().head(6).index
    category_by_cluster = pd.crosstab(
        top_df['cluster'], 
        top_df['category_primary'],
        normalize='index'
    )[top6_categories]
    
    # Plot category distribution
    plt.figure(figsize=(14, 8))
    category_by_cluster.plot(
        kind='bar',
        stacked=True,
        colormap='tab10',
        figsize=(14, 8)
    )
    plt.title('Business Category Distribution by Cluster (Top 6 Categories)')
    plt.xlabel('Cluster')
    plt.ylabel('Proportion')
    plt.legend(title='Business Category')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_by_cluster.png")
    
    # ====================== EXTRACT TEXT EXAMPLES ======================
    print("\n16. Extracting representative reviews from each cluster...")
    # Get sample reviews from each cluster
    sample_reviews = pd.DataFrame()
    for i in range(best_k):
        # Sort by review length to get more substantive reviews
        cluster_reviews = top_df[top_df['cluster'] == i].sort_values('review_length', ascending=False)
        # Get 3 examples from each sentiment category if available
        for sentiment in ['positive', 'neutral', 'negative']:
            sentiment_reviews = cluster_reviews[cluster_reviews['sentiment'] == sentiment].head(3)
            sample_reviews = pd.concat([sample_reviews, sentiment_reviews])
    
    # Save sample reviews to CSV (selected columns only)
    review_columns = ['cluster', 'sentiment', 'stars', 'category_primary', 'review_length', 'text']
    sample_reviews[review_columns].to_csv(f"{output_dir}/sample_reviews.csv", index=False)
    
    # ====================== SAVE CLUSTER PROFILES ======================
    print("\n17. Saving complete cluster profiles...")
    # Calculate additional statistics per cluster
    cluster_stats = pd.DataFrame()
    
    # Averages of key numeric features
    cluster_stats = pd.concat([cluster_stats, top_df.groupby('cluster')[base_features].mean()], axis=1)
    
    # Standard deviations of key features
    std_stats = top_df.groupby('cluster')[base_features].std()
    std_stats.columns = [f"{col}_std" for col in std_stats.columns]
    cluster_stats = pd.concat([cluster_stats, std_stats], axis=1)
    
    # Sentiment distribution
    sentiment_stats = pd.crosstab(top_df['cluster'], top_df['sentiment'], normalize='index')
    sentiment_stats.columns = [f"pct_{col}" for col in sentiment_stats.columns]
    cluster_stats = pd.concat([cluster_stats, sentiment_stats], axis=1)
    
    # Count and percentage
    counts = top_df['cluster'].value_counts().sort_index()
    percentages = counts / len(top_df) * 100
    cluster_stats['count'] = counts
    cluster_stats['percentage'] = percentages
    
    # Save to CSV
    cluster_stats.to_csv(f"{output_dir}/cluster_profiles_detailed.csv")
    
    # ====================== EVENT PLANNING SCORING ======================
    print("\n18. Creating business scoring for event planning...")
    # Group by business and calculate metrics
    business_metrics = top_df.groupby('business_id').agg({
        'stars': 'mean',
        'sentiment_value': 'mean',
        'review_length': 'mean',
        'recency_score': 'mean',
        'cluster': lambda x: x.value_counts().index[0],  # Most common cluster
        'category_primary': 'first'
    }).reset_index()
    
    # Calculate a composite score for event planning
    # Formula: 50% star rating + 30% sentiment + 20% recency
    business_metrics['event_score'] = (
        0.5 * business_metrics['stars'] / 5 +  # Normalize stars to 0-1
        0.3 * (business_metrics['sentiment_value'] + 1) / 2 +  # Normalize sentiment to 0-1
        0.2 * business_metrics['recency_score']
    )
    
    # Scale to 0-100 for easier interpretation
    business_metrics['event_score'] = business_metrics['event_score'] * 100
    
    # Add cluster label interpretation
    cluster_interpretations = {
        0: "Average reviews with mixed sentiment",
        1: "Detailed positive reviews",
        2: "Brief positive reviews",
        3: "Critical negative reviews"
    }
    business_metrics['cluster_description'] = business_metrics['cluster'].map(cluster_interpretations)
    
    # Sort by event score and save top businesses
    top_businesses = business_metrics.sort_values('event_score', ascending=False).head(100)
    top_businesses.to_csv(f"{output_dir}/top_event_businesses.csv", index=False)
    
    # ====================== FINAL SUMMARY ======================
    print("\n" + "="*50)
    print("CLUSTERING COMPLETE - SUMMARY")
    print("="*50)
    print(f"Number of businesses analyzed: {business_metrics['business_id'].nunique()}")
    print(f"Number of reviews analyzed: {len(top_df)}")
    print(f"Optimal number of clusters: {best_k}")
    
    print("\nCluster Sizes:")
    for cluster, size in cluster_sizes.items():
        print(f"Cluster {cluster}: {size} reviews ({size/len(top_df)*100:.1f}%)")
    
    print("\nCluster Interpretations:")
    for cluster, interpretation in cluster_interpretations.items():
        print(f"Cluster {cluster}: {interpretation}")
    
    print("\nTop Business Categories:")
    for category in top_categories:
        print(f"- {category}")
    
    print("\nResults saved to:", output_dir)
    print("="*50)

except Exception as e:
    print(f"\nError occurred: {str(e)}")
    import traceback
    traceback.print_exc()