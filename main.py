import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# --- Load and Preprocess Data ---
@st.cache_data
def load_data():
    train_df = pd.read_json('train.json')
    test_df = pd.read_json('test.json')
    train_df['ingredients_normalized'] = train_df['ingredients'].apply(normalize_ingredients)
    test_df['ingredients_normalized'] = test_df['ingredients'].apply(normalize_ingredients)
    return train_df, test_df

def normalize_ingredients(ingredients):
    return [ingredient.lower().replace(" ", "_").strip() for ingredient in ingredients]

@st.cache_resource
def compute_tfidf(train_df):
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train = tfidf_vectorizer.fit_transform(train_df['ingredients_normalized'].apply(" ".join))
    return X_train, tfidf_vectorizer

@st.cache_resource
def create_user_item_matrix(train_df):
    user_item_matrix = pd.DataFrame(np.random.randint(0, 5, size=(10, len(train_df))), 
                                    index=[f'user_{i}' for i in range(10)], 
                                    columns=train_df['id'])
    return user_item_matrix

# --- Recommendation Algorithms ---
def recommend_content_based(input_ingredients, X_train, train_df, tfidf_vectorizer, top_k=5):
    query_vec = tfidf_vectorizer.transform([" ".join(input_ingredients)])
    similarities = cosine_similarity(query_vec, X_train)
    top_indices = np.argsort(similarities.flatten())[::-1][:top_k]
    recommendations = train_df.iloc[top_indices][['id', 'cuisine', 'ingredients']]
    recommendations['similarity_score'] = similarities.flatten()[top_indices].astype(float)
    return recommendations

def recommend_collaborative(user_item_matrix, user_id, top_k=5):
    if user_id not in user_item_matrix.index:
        return pd.DataFrame({'id': [], 'cuisine': [], 'similarity_score': []})
    user_vector = user_item_matrix.loc[user_id].values.reshape(1, -1)
    similarities = cosine_similarity(user_vector, user_item_matrix)
    similar_users = np.argsort(similarities.flatten())[::-1][1:top_k+1]
    similar_items = user_item_matrix.iloc[similar_users].mean(axis=0)
    similar_items = similar_items[similar_items > 0]
    top_items = similar_items.sort_values(ascending=False).head(top_k).index
    recommendations = [{'id': item, 'cuisine': 'unknown', 'similarity_score': score}
                       for item, score in zip(top_items, similar_items[top_items])]
    return pd.DataFrame(recommendations)

def recommend_hybrid(input_ingredients, X_train, train_df, tfidf_vectorizer, user_item_matrix, user_id, top_k=5, alpha=0.7):
    content_recs = recommend_content_based(input_ingredients, X_train, train_df, tfidf_vectorizer, top_k=top_k)
    collab_recs = recommend_collaborative(user_item_matrix, user_id, top_k=top_k)
    
    # Merge and weight scores
    if not collab_recs.empty:
        collab_recs['id'] = collab_recs['id'].astype(int)
        combined = pd.merge(content_recs, collab_recs, on='id', how='outer', suffixes=('_content', '_collab')).fillna(0)
        combined['hybrid_score'] = alpha * combined['similarity_score_content'] + (1 - alpha) * combined['similarity_score_collab']
        combined = combined.sort_values('hybrid_score', ascending=False)
        combined['cuisine'] = combined['cuisine_content'].combine_first(combined['cuisine_collab'])
        return combined[['id', 'cuisine', 'ingredients', 'hybrid_score']]
    else:
        return content_recs

# --- EDA Visualizations ---
def plot_cuisine_distribution(train_df):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=train_df, x='cuisine', order=train_df['cuisine'].value_counts().index, palette='viridis', hue=None)
    plt.xticks(rotation=45)
    plt.title('Cuisine Distribution')
    st.pyplot(plt)

def plot_top_ingredients(train_df):
    all_ingredients = [ingredient for ingredients in train_df['ingredients_normalized'] for ingredient in ingredients]
    ingredient_counts = pd.Series(all_ingredients).value_counts()
    plt.figure(figsize=(12, 6))
    ingredient_counts.head(20).plot(kind='bar', color='skyblue')
    plt.title('Top 20 Ingredients')
    plt.xlabel('Ingredient')
    plt.ylabel('Count')
    st.pyplot(plt)

# --- NLP for Preferences ---
def analyze_reviews(reviews):
    sentiments = reviews.apply(lambda x: sia.polarity_scores(x)['compound'])
    preferences = {
        'positive_reviews': sentiments[sentiments > 0.5].count(),
        'negative_reviews': sentiments[sentiments < -0.5].count(),
        'neutral_reviews': sentiments[(sentiments <= 0.5) & (sentiments >= -0.5)].count()
    }
    return preferences

# --- Streamlit App Layout ---
def main():
    st.set_page_config(page_title="Recipe Recommendation System", layout="wide")
    st.title("🍽️ Recipe Recommendation System")
    st.sidebar.title("User Inputs")

    # Load and preprocess data
    train_df, test_df = load_data()
    X_train, tfidf_vectorizer = compute_tfidf(train_df)
    user_item_matrix = create_user_item_matrix(train_df)

    # Sidebar Inputs
    st.sidebar.subheader("Select Recommendation Algorithm")
    algo = st.sidebar.selectbox("Algorithm", ["Content-Based", "Collaborative Filtering", "Hybrid"], index=0)

    st.sidebar.subheader("Enter Ingredients")
    input_ingredients = st.sidebar.text_area("Ingredients (comma-separated)", "tomato, garlic, onion")
    input_ingredients = [ing.strip().lower() for ing in input_ingredients.split(",")]

    user_id = st.sidebar.text_input("Enter User ID for Collaborative/Hybrid", "user_1")

    st.sidebar.subheader("Toggle Analysis Sections")
    show_eda = st.sidebar.checkbox("Show EDA")
    show_recommendations = st.sidebar.checkbox("Show Recommendations", value=True)
    show_reviews = st.sidebar.checkbox("Show Review Analysis")

    # Main Content
    if show_eda:
        st.header("Exploratory Data Analysis")
        st.subheader("Cuisine Distribution")
        plot_cuisine_distribution(train_df)

        st.subheader("Top Ingredients")
        plot_top_ingredients(train_df)

    if show_recommendations:
        st.header("Recommended Recipes")
        if algo == "Content-Based":
            recommendations = recommend_content_based(input_ingredients, X_train, train_df, tfidf_vectorizer)
        elif algo == "Collaborative Filtering":
            recommendations = recommend_collaborative(user_item_matrix, user_id)
        else:
            recommendations = recommend_hybrid(input_ingredients, X_train, train_df, tfidf_vectorizer, user_item_matrix, user_id)

        if not recommendations.empty:
            st.write(recommendations)
        else:
            st.warning("No recommendations available for the given inputs!")

    if show_reviews:
        st.header("User Reviews Analysis")
        example_reviews = pd.Series([
            "I love spicy food and Italian pasta!",
            "Chinese recipes are too oily for me.",
            "Mexican tacos are my favorite."
        ])
        preferences = analyze_reviews(example_reviews)
        st.write("Preferences based on Reviews:")
        st.write(preferences)

    st.sidebar.subheader("About")
    st.sidebar.info("This app recommends recipes based on your inputs. Built with ❤️ using Streamlit.")

if __name__ == '__main__':
    main()
