# Import required libs
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,Dropout, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import constraints
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import load_model

# Suppress warnings
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")


## import datasets
ratings_df = pd.read_csv("C:/Users/Mouad/Desktop/Final_Year_Project/data/Ratings.csv")
books_df = pd.read_csv("C:/Users/Mouad/Desktop/Final_Year_Project/data/Books.csv")
users_df = pd.read_csv("C:/Users/Mouad/Desktop/Final_Year_Project/data/Users.csv")




print(ratings_df.columns)



# Merge datasets
df = ratings_df.merge(books_df, on='ISBN').merge(users_df, on='User-ID')



# Remove duplicates and unnecessary columns
df.drop_duplicates(subset=['User-ID', 'Book-Title'], keep='last', inplace=True)
df.drop(columns=['ISBN', 'Year-Of-Publication'], inplace=True)



# Filter out users and books with too few ratings
user_counts = df['User-ID'].value_counts()
book_counts = df['Book-Title'].value_counts()
df = df[df['User-ID'].isin(user_counts[user_counts >= 5].index)]
df = df[df['Book-Title'].isin(book_counts[book_counts >= 5].index)]

# Create user-item interaction matrix
interaction_matrix = df.pivot(index='User-ID', columns='Book-Title', values='Book-Rating').fillna(0)

# Encode user and item IDs
user_enc = LabelEncoder()
interaction_matrix.index = user_enc.fit_transform(interaction_matrix.index)

# Normalize ratings
scaler = MinMaxScaler()
interaction_matrix_normalized = scaler.fit_transform(interaction_matrix)


# Load the entire model
autoencoder = load_model('./AE_recommender/autoencoder_model.h5')



# Function to make book recommendations
def books_recommender(user_id, num_recommendations=5):
    # Encode the user ID
    encoded_user_id = user_enc.transform([user_id])[0]
    
    # Get the user's ratings
    user_ratings = interaction_matrix_normalized[encoded_user_id].reshape(1, -1)
    
    # Predict ratings for all books
    predicted_ratings = autoencoder.predict(user_ratings)
    
    # Convert predicted ratings to a Pandas Series
    predicted_ratings = pd.Series(predicted_ratings.flatten(), index=interaction_matrix.columns)
    
    # Get books the user has not rated
    user_original_ratings = interaction_matrix.loc[encoded_user_id]
    unrated_books = user_original_ratings[user_original_ratings == 0].index
    
    # Recommend the top n books
    recommendations = predicted_ratings.loc[unrated_books].sort_values(ascending=False).head(num_recommendations)
    
    return recommendations.index.tolist()

# Test the recommendation system
# user_id = 2012  # Replace with a valid user ID
# recommended_books = recommend_books(user_id, num_recommendations=5)
# print(f"Top 5 recommended books for user {user_id}:")
# print(recommended_books)
