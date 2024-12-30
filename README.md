
# Recipe Recommendation System

## Overview
This project involves building a personalized recipe recommendation system that suggests recipes based on user preferences, available ingredients, and constraints (e.g., time to cook or dietary restrictions). The system employs three recommendation algorithms: content-based filtering, collaborative filtering, and a hybrid approach combining both techniques. It also features an interactive Streamlit application to enable users to explore recommendations, analyze data, and understand user preferences.

---

## Approach
### 1. Data Understanding and Preprocessing
- **Dataset**: Utilized JSON-formatted datasets for training and testing, each containing recipe IDs, ingredients, and cuisine labels.
- **Normalization**: Ingredients were normalized (e.g., converting to lowercase and replacing spaces with underscores) to ensure consistency across the dataset.
- **TF-IDF Vectorization**: Ingredients were transformed into TF-IDF vectors to represent textual data numerically for content-based filtering.
- **User-Item Matrix**: Simulated a user-item interaction matrix to support collaborative filtering by generating random user interactions.

### 2. Recommendation Algorithms
#### a. Content-Based Filtering:
- **Method**: Calculated cosine similarity between user-provided ingredients and recipes in the dataset using TF-IDF vectors.
- **Output**: Recommended recipes with the highest similarity scores.

#### b. Collaborative Filtering:
- **Method**: Used cosine similarity to find similar users based on the user-item matrix and aggregated item scores for recommendations.
- **Fallback**: Ensured recommendations even when user interactions were sparse by merging with default content-based recommendations.

#### c. Hybrid Approach:
- **Method**: Combined scores from content-based and collaborative filtering algorithms using a weighted average.
- **Customization**: Enabled users to control the weight (alpha) between the two approaches via the Streamlit interface.

### 3. Exploratory Data Analysis (EDA)
- Visualized cuisine distributions to understand the balance across categories.
- Highlighted the most frequent ingredients to provide insights into common patterns in the data.

### 4. Sentiment Analysis for Review Analysis
- Applied the VADER sentiment analysis tool to analyze user reviews and classify them as positive, negative, or neutral.

---

## Challenges and Solutions
### Challenge 1: Handling Sparse User Interactions
- **Issue**: Collaborative filtering suffered from sparse user-item interactions.
- **Solution**: Simulated a synthetic user-item matrix and incorporated fallback mechanisms to rely on content-based filtering when necessary.

### Challenge 2: Merging Content-Based and Collaborative Recommendations
- **Issue**: Merging recommendations while ensuring data consistency and handling missing values.
- **Solution**: Used pandas functions with careful handling of missing values and ensured scores were normalized before merging.

### Challenge 3: Ingredient Variability
- **Issue**: Variations in ingredient names (e.g., "tomato" vs. "tomatoes") impacted recommendation accuracy.
- **Solution**: Normalized ingredients by converting to lowercase, stripping spaces, and replacing them with underscores.

---

## Ideas for Improvement
1. **Enhanced Collaborative Filtering**:
   - Incorporate implicit feedback (e.g., clicks or time spent on a recipe) for better recommendations.
   - Use advanced techniques like matrix factorization (e.g., SVD) or deep learning-based collaborative filtering models.

2. **Real-Time Data Integration**:
   - Allow users to add custom ingredients or preferences in real time and dynamically update recommendations.

3. **Improved Sentiment Analysis**:
   - Use fine-tuned transformer models (e.g., BERT) for more accurate review analysis and preference extraction.

4. **Scalability**:
   - Migrate the backend to a cloud-based infrastructure to handle larger datasets and real-time recommendations.

5. **User Personalization**:
   - Implement a user profile feature to save and refine recommendations based on historical interactions.

6. **Mobile Application**:
   - Extend the application to a mobile platform for broader accessibility and convenience.

---

## Conclusion
The Recipe Recommendation System is a robust, modular solution that combines multiple recommendation techniques to cater to diverse user needs. It provides a scalable foundation for further enhancements, such as real-time data integration, advanced recommendation algorithms, and enhanced user personalization.
