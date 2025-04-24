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
output_dir = "enhanced_event_planning_system"
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
    del review_df  
    gc.collect()
    print(f"   Review sample loaded: {len(review_sample)} rows")
    
    print("\n2. Preprocessing data...")
    merged_df = pd.merge(review_sample, business_df, on='business_id', how='inner', suffixes=('_review', '_business'))
    del review_sample  
    gc.collect()
    print(f"   Merged data: {len(merged_df)} rows")
    
    
    merged_df.dropna(subset=['text', 'stars_review', 'categories'], inplace=True)
    
    # Convert date to datetime and extract time features
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['days_since'] = (pd.Timestamp.now() - merged_df['date']).dt.days
    merged_df['recency_score'] = 1 / np.log1p(merged_df['days_since'] + 1)  # Higher score for more recent reviews
    
    # Create sentiment scores - Simple method based on ratings
    merged_df['sentiment_value'] = np.select(
        [merged_df['stars_review'] <= 2, merged_df['stars_review'] == 3, merged_df['stars_review'] > 3],
        [-1, 0, 1]
    )
    merged_df['sentiment'] = np.select(
        [merged_df['stars_review'] <= 2, merged_df['stars_review'] == 3, merged_df['stars_review'] > 3],
        ['negative', 'neutral', 'positive']
    )
    
    # ====================== ENHANCED TEXT PROCESSING ======================
    print("\n3. Performing enhanced text processing...")
    
    # Calculate review length for feature engineering
    merged_df['review_length'] = merged_df['text'].str.len()
    

    print("   Extracting keywords from reviews...")
    
    # Define sets of keywords related to different event types
    event_keywords = {
        'Wedding': ['wedding', 'ceremony', 'bride', 'groom', 'reception', 'marriage', 'celebrate', 'elegant', 'romantic', 'formal'],
        'Corporate': ['business', 'meeting', 'conference', 'corporate', 'professional', 'presentation', 'client', 'office', 'formal', 'network'],
        'Birthday': ['birthday', 'celebrate', 'party', 'fun', 'cake', 'gift', 'happy', 'surprise', 'friend', 'family'],
        'Casual Gathering': ['casual', 'hangout', 'chill', 'relax', 'friends', 'drinks', 'social', 'informal', 'comfortable', 'group']
    }
    
    # Function to check if review mentions events
    def contains_event_keywords(text, event_type):
        if not isinstance(text, str):
            return 0
        text = text.lower()
        count = sum(1 for keyword in event_keywords[event_type] if keyword in text)
        return min(count / len(event_keywords[event_type]), 1.0)  # Normalize to 0-1
    
    # Add event keyword scores to dataframe
    for event_type in event_keywords:
        col_name = f'mentions_{event_type.lower().replace(" ", "_")}'
        merged_df[col_name] = merged_df['text'].apply(
            lambda x: contains_event_keywords(x, event_type)
        )
    
    # ====================== CATEGORY PROCESSING ======================
    print("\n4. Processing business categories...")
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
    print("\n5. Engineering improved features for clustering...")
    
    # Calculate additional metrics
    if 'useful' in merged_df.columns:
        merged_df['useful_score'] = merged_df['useful'].fillna(0) / merged_df['useful'].max()
    else:
        merged_df['useful_score'] = 0
    
    # Calculate event mention scores (average of all event types)
    event_mention_cols = [f'mentions_{event_type.lower().replace(" ", "_")}' for event_type in event_keywords]
    merged_df['event_mentions_score'] = merged_df[event_mention_cols].mean(axis=1)
    
    # Define enhanced features for clustering
    features = [
        'stars_review',          # Rating given by customers
        'sentiment_value',       # Derived sentiment
        'review_length',         # Length of review as proxy for detail
        'recency_score',         # How recent the review is
        'useful_score',          # How useful others found the review
        'event_mentions_score'   # How often review mentions event-related terms
    ]
    
    # ====================== BUSINESS METRICS CALCULATION ======================
    print("\n6. Calculating enhanced business metrics...")
    # Group by business_id to get metrics for each business
    business_metrics = merged_df.groupby('business_id').agg({
        'stars_review': 'mean',
        'sentiment_value': 'mean',
        'review_length': 'mean',
        'recency_score': 'mean',
        'useful_score': 'mean',
        'primary_category': 'first',
        'categories': 'first',
        'name': 'first',  
        'stars_business': 'first'  
    }).reset_index()
    
    # Add category flags to business metrics
    for category in event_categories:
        cat_col = f'is_{category.lower().replace(" & ", "_").replace(" ", "_")}'
        if cat_col in merged_df.columns:
            business_metrics[cat_col] = merged_df.groupby('business_id')[cat_col].max().values
    
    # Add event mention scores to business metrics
    for event_type in event_keywords:
        col_name = f'mentions_{event_type.lower().replace(" ", "_")}'
        business_metrics[col_name] = merged_df.groupby('business_id')[col_name].mean().values
    
    # ====================== IMPROVED CLUSTERING ======================
    print("\n7. Performing improved clustering on reviews...")
    # Prepare data for clustering
    X = merged_df[features].copy()
    
    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Save PCA projection for visualization
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['business_id'] = merged_df['business_id'].values
    pca_df['sentiment'] = merged_df['sentiment_value'].values
    pca_df.to_csv(f"{output_dir}/pca_projection.csv", index=False)
    
    # Apply K-means clustering with optimal k
    # Try different k values
    k_range = range(2, 8)
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score
        try:
            score = silhouette_score(X_scaled, cluster_labels)
            silhouette_scores.append(score)
            print(f"   Silhouette score for k={k}: {score:.4f}")
        except:
            silhouette_scores.append(-1)
    
    # Find optimal k
    optimal_k = k_range[silhouette_scores.index(max(silhouette_scores))]
    print(f"   Optimal number of clusters: {optimal_k}")
    
    # Apply K-means with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    merged_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Calculate cluster profiles
    cluster_profiles = merged_df.groupby('cluster')[features + ['sentiment']].agg({
        'stars_review': 'mean',
        'sentiment_value': 'mean',
        'review_length': 'mean',
        'recency_score': 'mean',
        'useful_score': 'mean',
        'event_mentions_score': 'mean',
        'sentiment': lambda x: x.value_counts().index[0]
    })
    
    # Map businesses to their dominant cluster
    business_cluster_counts = merged_df.groupby(['business_id', 'cluster']).size().unstack().fillna(0)
    business_metrics['dominant_cluster'] = business_cluster_counts.idxmax(axis=1)
    
    # ====================== IMPROVED EVENT PLANNING SCORES ======================
    print("\n8. Creating improved event planning scoring system...")
    
    # Create a composite score for each business with more weight on sentiment
    # Updated formula: 30% average rating + 30% sentiment + 20% recency + 10% usefulness + 10% event mentions
    business_metrics['base_score'] = (
        0.30 * business_metrics['stars_review'] / 5 +  # Normalize to 0-1
        0.30 * (business_metrics['sentiment_value'] + 1) / 2 +  # Normalize to 0-1
        0.20 * business_metrics['recency_score'] +
        0.10 * business_metrics['useful_score']
    ) * 100  # Scale to 0-100
    
    # Create event-specific scores based on category relevance and event mentions
    for event_type in event_types:
        # Calculate category match score (how well the business matches the event type)
        category_match = np.zeros(len(business_metrics))
        
        for category in event_types[event_type]:
            cat_col = f'is_{category.lower().replace(" & ", "_").replace(" ", "_")}'
            if cat_col in business_metrics.columns:
                # Add weight based on category position in the relevant_categories list
                weight = 1 - 0.1 * event_types[event_type].index(category)  # Higher weight for more important categories
                category_match += business_metrics[cat_col] * weight
        
        # Normalize category match score
        max_possible = sum([1 - 0.1 * i for i in range(len(event_types[event_type]))])
        category_match = category_match / max_possible
        
        # Get event mention score
        event_mention_col = f'mentions_{event_type.lower().replace(" ", "_")}'
        event_mention_score = business_metrics[event_mention_col]
        
        # Calculate event-specific score: 60% base score + 30% category match + 10% event mentions
        business_metrics[f'{event_type.lower()}_score'] = (
            0.60 * business_metrics['base_score'] +
            0.30 * category_match * 100 +  # Scale to 0-100
            0.10 * event_mention_score * 100  # Scale to 0-100
        )
    
    # ====================== IMPROVED RECOMMENDATION SYSTEM ======================
    print("\n9. Building improved recommendation system...")
    
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
    
    # ====================== REVIEW SUMMARIZATION ======================
    print("\n10. Generating business insights...")
    
    # Function to generate business insights based on event type
    def generate_business_insight(business_id, event_type):
        """Generate insights about why a business is good for a specific event type"""
        try:
            # Get business information
            business_info = business_metrics[business_metrics['business_id'] == business_id].iloc[0]
            business_name = business_info['name']
            
            # Get relevant category and score information
            primary_category = business_info['primary_category']
            event_score = business_info[f'{event_type.lower()}_score']
            stars = business_info['stars_business']
            
            # Event-specific insights
            if event_type == 'Wedding':
                return f"{business_name} is recommended for weddings with a score of {event_score:.1f}/100. As a {primary_category} with {stars:.1f} stars, it's particularly suitable for elegant wedding celebrations."
            elif event_type == 'Corporate':
                return f"{business_name} is ideal for corporate events with a score of {event_score:.1f}/100. This {primary_category} (rated {stars:.1f} stars) offers a professional setting for business gatherings."
            elif event_type == 'Birthday':
                return f"{business_name} scores {event_score:.1f}/100 for birthday celebrations. This {stars:.1f}-star {primary_category} is perfect for fun birthday gatherings."
            else:  # Casual Gathering
                return f"{business_name} is great for casual gatherings with friends, scoring {event_score:.1f}/100. This {stars:.1f}-star {primary_category} offers a relaxed atmosphere."
        except (IndexError, KeyError):
            return f"This business is recommended for {event_type} events based on review analysis."
    
    # Add insights to recommendations
    for event_type in event_recommendations:
        for category in event_recommendations[event_type]:
            for i, business in enumerate(event_recommendations[event_type][category]):
                business_id = business['business_id']
                event_recommendations[event_type][category][i]['insight'] = generate_business_insight(business_id, event_type)
    
    # Save updated recommendations
    with open(f"{output_dir}/event_recommendations_with_insights.json", "w") as f:
        json.dump(event_recommendations, f, indent=2)
    
    # ====================== CONTEXTUAL RECOMMENDATION FEATURES ======================
    print("\n11. Adding contextual recommendation features...")
    
    # Define time period relevance for businesses
    time_periods = {
        'morning': ['Breakfast & Brunch', 'Coffee & Tea', 'Bakeries'],
        'afternoon': ['Lunch', 'Coffee & Tea', 'Desserts', 'Shopping'],
        'evening': ['Dinner', 'Bars', 'Nightlife', 'Entertainment'],
        'weekend': ['Brunch', 'Outdoor', 'Entertainment', 'Nightlife'],
        'weekday': ['Lunch', 'Coffee & Tea', 'Dinner']
    }
    
    # Function to determine time relevance score
    def get_time_relevance(business_categories, time_period):
        """Calculate time period relevance score based on business categories"""
        if not isinstance(business_categories, str):
            return 0.5  # Default if no categories
            
        relevance = 0
        period_categories = time_periods.get(time_period, [])
        
        # Check for matches between business categories and time period categories
        for category in period_categories:
            if category in business_categories:
                relevance += 1
                
        # Normalize
        if period_categories:
            relevance = min(1.0, relevance / len(period_categories))
        else:
            relevance = 0.5
            
        return relevance
    
    # Add time relevance to businesses
    for time_period in time_periods:
        col_name = f'{time_period}_relevance'
        business_metrics[col_name] = business_metrics['categories'].apply(
            lambda x: get_time_relevance(x, time_period)
        )
    
    # Save enriched business metrics
    business_metrics.to_csv(f"{output_dir}/enriched_business_metrics.csv", index=False)
    
    # ====================== EVENT PLAN GENERATOR ======================
    print("\n12. Creating enhanced event plan generator...")
    
    def generate_event_plans(event_type, context=None, num_plans=3):
        """Generate multiple event plans for a specific event type with contextual awareness"""
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
                    
                    venue_entry = {
                        "category": category,
                        "name": business["name"],
                        "rating": business["stars_business"],
                        "score": round(business[f"{event_type.lower()}_score"], 1),
                        "insight": business.get("insight", "")
                    }
                    
                    # Add contextual info if provided
                    if context:
                        if 'time_of_day' in context and context['time_of_day'] in time_periods:
                            time_period = context['time_of_day']
                            time_col = f'{time_period}_relevance'
                            if time_col in business_metrics.columns:
                                business_id = business["business_id"]
                                time_relevance = business_metrics[business_metrics['business_id'] == business_id][time_col].values[0]
                                venue_entry["time_relevance"] = round(time_relevance * 100)
                                
                                if time_relevance > 0.7:
                                    venue_entry["time_context"] = f"Highly recommended for {time_period} events."
                                elif time_relevance > 0.3:
                                    venue_entry["time_context"] = f"Suitable for {time_period} events."
                    
                    plan["venues"].append(venue_entry)
            
            # Calculate overall plan score (average of venue scores)
            if plan["venues"]:
                plan["overall_score"] = round(
                    sum(venue["score"] for venue in plan["venues"]) / len(plan["venues"]),
                    1
                )
                plans.append(plan)
        
        return plans
    
    # Generate plans for each event type with context and save
    all_plans = {}
    contexts = {
        'Wedding': {'time_of_day': 'evening'},
        'Corporate': {'time_of_day': 'weekday'},
        'Birthday': {'time_of_day': 'weekend'},
        'Casual Gathering': {'time_of_day': 'evening'}
    }
    
    for event_type in event_types:
        all_plans[event_type] = generate_event_plans(
            event_type, 
            context=contexts.get(event_type)
        )
    
    # Save all plans to JSON
    with open(f"{output_dir}/enhanced_event_plans.json", "w") as f:
        json.dump(all_plans, f, indent=2)
    
    # ====================== ENHANCED RECOMMENDATION FUNCTION ======================
    print("\n13. Creating enhanced recommendation function for LLM integration...")
    
    def get_event_recommendations(event_type, preferences=None, context=None):
        """
        Enhanced function for LLM to generate event recommendations
        
        Args:
            event_type: Type of event (Wedding, Corporate, Birthday, Casual Gathering)
            preferences: Optional dict with preferences like budget, location, etc.
            context: Optional dict with contextual info like time, season, weather
            
        Returns:
            Dictionary with recommended plans
        """
        event_type = event_type.title()  # Standardize capitalization
        
        # Handle case where event type is not in our predefined types
        if event_type not in event_types:
            closest_match = min(event_types.keys(), key=lambda x: len(set(x.lower().split()) & set(event_type.lower().split())))
            event_type = closest_match
        
        # Get the plans for this event type
        if event_type in all_plans and all_plans[event_type]:
            plans = all_plans[event_type]
            
            # Apply contextual filtering if provided
            if context:
                # Adjust plans based on time of day
                time_of_day = context.get('time_of_day')
                if time_of_day in time_periods:
                    for plan in plans:
                        plan['time_context'] = f"This plan is optimized for {time_of_day} events."
            
            # Format the response for the LLM
            response = {
                "event_type": event_type,
                "plans": plans
            }
            
            # Add explanation of scores
            response["explanation"] = (
                f"Each {event_type.lower()} plan includes top-rated venues across different categories. "
                f"Scores are calculated based on customer ratings, sentiment analysis, review recency, "
                f"and relevance to {event_type.lower()} events. "
                f"We've analyzed reviews to identify which venues are most suitable for your event type."
            )
            
            # Add explanation for contextual factors if provided
            if context:
                contextual_factors = []
                if context.get('time_of_day'):
                    contextual_factors.append(f"time of day ({context['time_of_day']})")
                
                if contextual_factors:
                    response["contextual_explanation"] = (
                        f"These recommendations are optimized for the following contextual factors: "
                        f"{', '.join(contextual_factors)}."
                    )
            
            return response
        else:
            return {"error": f"No recommendations available for {event_type} events."}
    
    # Save example output with context
    example_output = get_event_recommendations(
        "Wedding", 
        context={'time_of_day': 'evening'}
    )
    with open(f"{output_dir}/example_recommendation.json", "w", encoding="utf-8") as f:
        json.dump(example_output, f, indent=2, ensure_ascii=False)
    
    # ====================== SUMMARY ======================
    print("\n" + "="*50)
    print("ENHANCED EVENT PLANNING RECOMMENDATION SYSTEM COMPLETE")
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
            if 'insight' in venue:
                print(f"    Insight: {venue['insight']}")
    
   

except Exception as e:
    import traceback
    print(f"\nError occurred: {str(e)}")
    traceback.print_exc()