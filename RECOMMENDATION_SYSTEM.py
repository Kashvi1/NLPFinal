import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import gc
import json
import random
from collections import Counter, defaultdict

# Set up output directory
output_dir = "event_planning_system"
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {output_dir}")

try:
    # ====================== DATA LOADING ======================
    print("\n1. Loading data...")
    business_file_path = '/Users/omniaabouhassan/Desktop/NLPFinal/cleaned_yelp_business_data (2).csv'
    review_file_path = '/Users/omniaabouhassan/Desktop/NLPFinal/cleaned_yelp_review_data.csv'
    
    # Check if files exist
    for file_path in [business_file_path, review_file_path]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load business data with all relevant columns for event planning
    business_df = pd.read_csv(business_file_path, 
                             dtype={'business_id': 'str'})
    print(f"   Business data loaded: {len(business_df)} rows")
    print(f"   Business columns: {', '.join(business_df.columns)}")
    
    # Extract only necessary columns to save memory
    needed_cols = ['business_id', 'stars', 'text', 'date']
    if 'useful' in business_df.columns:
        needed_cols.append('useful')
    
    # Load 15% sample of reviews to manage memory
    print("   Loading review sample...")
    review_df = pd.read_csv(review_file_path, 
                           dtype={'business_id': 'str', 'stars': 'int8'},
                           usecols=needed_cols)
    sample_size = int(len(review_df) * 0.15)
    review_sample = review_df.sample(n=sample_size, random_state=42)
    del review_df  # Free memory
    gc.collect()
    print(f"   Review sample loaded: {len(review_sample)} rows")
    
    # ====================== DATA MERGING & PREPROCESSING ======================
    print("\n2. Preprocessing data...")
    merged_df = pd.merge(review_sample, business_df, on='business_id', how='inner', suffixes=('_review', '_business'))
    del review_sample  # Free memory
    gc.collect()
    print(f"   Merged data: {len(merged_df)} rows")
    
    # Clean and prepare data
    merged_df.dropna(subset=['text', 'stars_review', 'categories'], inplace=True)
    
    # Convert date to datetime and extract time features
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['days_since'] = (pd.Timestamp.now() - merged_df['date']).dt.days
    merged_df['recency_score'] = 1 / np.log1p(merged_df['days_since'] + 1)  # Higher score for more recent reviews
    
    # Create sentiment scores
    merged_df['sentiment_value'] = np.select(
        [merged_df['stars_review'] <= 2, merged_df['stars_review'] == 3, merged_df['stars_review'] > 3],
        [-1, 0, 1]
    )
    merged_df['sentiment'] = np.select(
        [merged_df['stars_review'] <= 2, merged_df['stars_review'] == 3, merged_df['stars_review'] > 3],
        ['negative', 'neutral', 'positive']
    )
    
    # ====================== CATEGORY PROCESSING ======================
    print("\n3. Processing business categories...")
    # Extract all categories
    all_categories = []
    for cats in merged_df['categories'].dropna():
        if isinstance(cats, str):
            all_categories.extend([c.strip() for c in cats.split(',')])
    
    # Find top 25 categories (more than before to get variety for events)
    category_counts = Counter(all_categories)
    top_categories = [cat for cat, count in category_counts.most_common(25)]
    print(f"   Top categories: {', '.join(top_categories[:10])}...")
    
    # Map each business to its primary category and all categories
    merged_df['primary_category'] = merged_df['categories'].str.split(',').str[0].str.strip()
    
    # Create category flags for relevant event planning categories
    event_categories = [
        'Restaurants', 'Bars', 'Venues & Event Spaces', 'Hotels',
        'Caterers', 'Food', 'Nightlife', 'Event Planning & Services',
        'Photographers', 'DJs', 'Bakeries', 'Desserts'
    ]
    
    # Create flags for event-related categories
    for category in event_categories:
        if category in top_categories:
            merged_df[f'is_{category.lower().replace(" & ", "_").replace(" ", "_")}'] = \
                merged_df['categories'].str.contains(category, case=False, na=False).astype(int)
    
    # Define event types and their relevant categories
    event_types = {
        'Wedding': ['Venues & Event Spaces', 'Caterers', 'Photographers', 'DJs', 'Bakeries', 'Hotels', 'Restaurants'],
        'Corporate': ['Venues & Event Spaces', 'Caterers', 'Hotels', 'Restaurants', 'Event Planning & Services'],
        'Birthday': ['Restaurants', 'Bars', 'Venues & Event Spaces', 'Bakeries', 'Desserts', 'Nightlife'],
        'Casual Gathering': ['Restaurants', 'Bars', 'Food', 'Nightlife']
    }
    
    # ====================== FEATURE ENGINEERING ======================
    print("\n4. Engineering features for clustering...")
    # Calculate review length
    merged_df['review_length'] = merged_df['text'].str.len()
    
    # Calculate additional metrics
    if 'useful' in merged_df.columns:
        merged_df['useful_score'] = merged_df['useful'].fillna(0) / merged_df['useful'].max()
    else:
        merged_df['useful_score'] = 0
    
    # Define features for clustering
    features = [
        'stars_review',          # Rating given by customers
        'sentiment_value',       # Derived sentiment
        'review_length',         # Length of review as proxy for detail
        'recency_score',         # How recent the review is
        'useful_score'           # How useful others found the review
    ]
    
    # ====================== BUSINESS METRICS CALCULATION ======================
    print("\n5. Calculating business metrics for recommendations...")
    # Group by business_id to get metrics for each business
    business_metrics = merged_df.groupby('business_id').agg({
        'stars_review': 'mean',
        'sentiment_value': 'mean',
        'review_length': 'mean',
        'recency_score': 'mean',
        'useful_score': 'mean',
        'primary_category': 'first',
        'categories': 'first',
        'name': 'first',  # Assuming 'name' column exists
        'stars_business': 'first'  # Overall business rating
    }).reset_index()
    
    # Add category flags to business metrics
    for category in event_categories:
        cat_col = f'is_{category.lower().replace(" & ", "_").replace(" ", "_")}'
        if cat_col in merged_df.columns:
            business_metrics[cat_col] = merged_df.groupby('business_id')[cat_col].max().values
    
    # ====================== CLUSTERING ======================
    print("\n6. Performing clustering on reviews...")
    # Prepare data for clustering
    X = merged_df[features].copy()
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply K-means clustering with k=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    merged_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster profiles
    cluster_profiles = merged_df.groupby('cluster')[features + ['sentiment']].agg({
        'stars_review': 'mean',
        'sentiment_value': 'mean',
        'review_length': 'mean',
        'recency_score': 'mean',
        'useful_score': 'mean',
        'sentiment': lambda x: x.value_counts().index[0]
    })
    
    # Map businesses to their dominant cluster
    business_cluster_counts = merged_df.groupby(['business_id', 'cluster']).size().unstack().fillna(0)
    business_metrics['dominant_cluster'] = business_cluster_counts.idxmax(axis=1)
    
    # ====================== EVENT PLANNING SCORES ======================
    print("\n7. Creating event planning scoring system...")
    
    # Create a composite score for each business
    # Base score formula: 40% average rating + 30% sentiment + 20% recency + 10% usefulness
    business_metrics['base_score'] = (
        0.4 * business_metrics['stars_review'] / 5 +  # Normalize to 0-1
        0.3 * (business_metrics['sentiment_value'] + 1) / 2 +  # Normalize to 0-1
        0.2 * business_metrics['recency_score'] +
        0.1 * business_metrics['useful_score']
    ) * 100  # Scale to 0-100
    
    # Create event-specific scores based on category relevance
    for event_type, relevant_categories in event_types.items():
        # Calculate category match score (how well the business matches the event type)
        category_match = np.zeros(len(business_metrics))
        
        for category in relevant_categories:
            cat_col = f'is_{category.lower().replace(" & ", "_").replace(" ", "_")}'
            if cat_col in business_metrics.columns:
                # Add weight based on category position in the relevant_categories list
                weight = 1 - 0.1 * relevant_categories.index(category)  # Higher weight for more important categories
                category_match += business_metrics[cat_col] * weight
        
        # Normalize category match score
        max_possible = sum([1 - 0.1 * i for i in range(len(relevant_categories))])
        category_match = category_match / max_possible
        
        # Calculate event-specific score: 70% base score + 30% category match
        business_metrics[f'{event_type.lower()}_score'] = (
            0.7 * business_metrics['base_score'] +
            0.3 * category_match * 100  # Scale to 0-100
        )
    
    # ====================== EVENT PLANNING RECOMMENDATION SYSTEM ======================
    print("\n8. Building recommendation system...")
    
    # Create a dictionary to store top businesses for each category by event type
    event_recommendations = {}
    
    for event_type in event_types:
        event_score = f'{event_type.lower()}_score'
        event_recommendations[event_type] = {}
        
        for category in event_types[event_type]:
            cat_col = f'is_{category.lower().replace(" & ", "_").replace(" ", "_")}'
            
            if cat_col in business_metrics.columns:
                # Filter businesses in this category
                category_businesses = business_metrics[business_metrics[cat_col] == 1].copy()
                
                if len(category_businesses) > 0:
                    # Sort by event-specific score
                    top_businesses = category_businesses.sort_values(event_score, ascending=False).head(5)
                    
                    # Store top businesses for this category
                    event_recommendations[event_type][category] = top_businesses[
                        ['business_id', 'name', 'stars_business', event_score, 'primary_category']
                    ].to_dict('records')
    
    # Save recommendations to JSON
    with open(f"{output_dir}/event_recommendations.json", "w") as f:
        json.dump(event_recommendations, f, indent=2)
    
    # ====================== EVENT PLAN GENERATOR ======================
    print("\n9. Creating event plan generator...")
    
    def generate_event_plans(event_type, num_plans=3):
        """Generate multiple event plans for a specific event type"""
        if event_type not in event_recommendations:
            return f"Sorry, we don't have data for {event_type} events."
        
        plans = []
        categories = event_types[event_type]
        
        # For each plan, select different top businesses from each category
        for plan_num in range(num_plans):
            plan = {
                "name": f"{event_type} Plan {plan_num + 1}",
                "venues": []
            }
            
            # Select businesses from each relevant category
            for category in categories:
                if category in event_recommendations[event_type] and len(event_recommendations[event_type][category]) > 0:
                    # Select different businesses for different plans when possible
                    idx = min(plan_num, len(event_recommendations[event_type][category]) - 1)
                    business = event_recommendations[event_type][category][idx]
                    
                    plan["venues"].append({
                        "category": category,
                        "name": business["name"],
                        "rating": business["stars_business"],
                        "score": round(business[f"{event_type.lower()}_score"], 1)
                    })
            
            # Calculate overall plan score (average of venue scores)
            if plan["venues"]:
                plan["overall_score"] = round(
                    sum(venue["score"] for venue in plan["venues"]) / len(plan["venues"]),
                    1
                )
                plans.append(plan)
        
        return plans
    
    # Generate plans for each event type and save
    all_plans = {}
    for event_type in event_types:
        all_plans[event_type] = generate_event_plans(event_type)
    
    # Save all plans to JSON
    with open(f"{output_dir}/event_plans.json", "w") as f:
        json.dump(all_plans, f, indent=2)
    
    # ====================== SENTIMENT ANALYSIS BY CATEGORY ======================
    print("\n10. Analyzing sentiment by business category...")
    
    # Calculate sentiment distribution by category
    sentiment_by_category = pd.crosstab(
        merged_df['primary_category'], 
        merged_df['sentiment'], 
        normalize='index'
    ).reset_index()
    
    # Save to CSV
    sentiment_by_category.to_csv(f"{output_dir}/sentiment_by_category.csv", index=False)
    
    # ====================== CREATE RECOMMENDATION FUNCTION ======================
    print("\n11. Creating recommendation function for LLM integration...")
    
    def get_event_recommendations(event_type, preferences=None):
        """
        Function for LLM to generate event recommendations
        
        Args:
            event_type: Type of event (Wedding, Corporate, Birthday, Casual Gathering)
            preferences: Optional dict with preferences like budget, location, etc.
            
        Returns:
            Dictionary with three recommended plans
        """
        event_type = event_type.title()  # Standardize capitalization
        
        # Handle case where event type is not in our predefined types
        if event_type not in event_types:
            closest_match = min(event_types.keys(), key=lambda x: len(set(x.lower().split()) & set(event_type.lower().split())))
            event_type = closest_match
            
        # Get the plans for this event type
        if event_type in all_plans and all_plans[event_type]:
            plans = all_plans[event_type]
            
            # Format the response for the LLM
            response = {
                "event_type": event_type,
                "plans": plans
            }
            
            # Add explanation of scores
            response["explanation"] = (
                f"Each {event_type.lower()} plan includes top-rated venues across different categories. "
                f"Scores are calculated based on customer ratings, sentiment analysis, review recency, "
                f"and relevance to {event_type.lower()} events."
            )
            
            return response
        else:
            return {"error": f"No recommendations available for {event_type} events."}
    
    # Save example output for testing
    example_output = get_event_recommendations("Wedding")
    with open(f"{output_dir}/example_recommendation.json", "w", encoding="utf-8") as f:
        json.dump(example_output, f, indent=2, ensure_ascii=False)
    
    # ====================== SUMMARY ======================
    print("\n" + "="*50)
    print("EVENT PLANNING RECOMMENDATION SYSTEM COMPLETE")
    print("="*50)
    print(f"Processed {len(merged_df)} reviews for {len(business_metrics)} businesses")
    print(f"Created recommendation systems for: {', '.join(event_types.keys())}")
    print(f"All results saved to: {output_dir}")
    
    # Print example recommendation
    print("\nExample wedding recommendation:")
    for i, plan in enumerate(example_output["plans"]):
        print(f"\nPlan {i+1} (Score: {plan.get('overall_score', 'N/A')}):")
        for venue in plan["venues"]:
            print(f"  • {venue['category']}: {venue['name']} (Rating: {venue['rating']}, Score: {venue['score']})")
    
    print("\n" + "="*50)
    print("To use this system with an LLM, load the event_plans.json file")
    print("and implement the get_event_recommendations function.")
    print("="*50)

except Exception as e:
    import traceback
    print(f"\nError occurred: {str(e)}")
    traceback.print_exc()