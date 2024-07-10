import pandas as pd
from surprise import SVD
import joblib

# Load the trained model
model_SVD = joblib.load('./SVD_recommender/best_svd_model.pkl')

# Load complete dataset
complete_df = pd.read_csv("./SVD_recommender/complete.csv")

# Define function to recommend books for a given user
def recommend_books(user_id, n=5):
    all_books = complete_df["Book-Title"].unique()
    rated_books = complete_df[complete_df["User-ID"] == user_id]["Book-Title"].values
    books_to_predict = [book for book in all_books if book not in rated_books]

    predictions = []
    for book in books_to_predict:
        pred = model_SVD.predict(user_id, book)
        predictions.append((book, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    # Get top N recommendations
    top_n = predictions[:n]
    rec_books = [title for title, rating in top_n]
    return rec_books

# # Load books data
# books_df = pd.read_csv("./books.csv")

# # Extract necessary columns
# df_imag_url = books_df[["Book-Title", "Image-URL-L"]]

# Recommend books for a specific user
# user_id = 2012
# recommended_books = recommend_books(user_id=user_id, n=5)

# print(recommended_books)

# # Filter and remove duplicates
# df_rec2 = df_imag_url.loc[df_imag_url["Book-Title"].isin([title for title, rating in recommended_books])]
# df_rec2 = df_rec2.drop_duplicates(subset=["Book-Title"], keep="first")
# df_rec2 = df_rec2.drop_duplicates(subset=["Book-Title", "Image-URL-L"], keep="first")

# # Print the recommended books
# print(f"Top 5 recommended books for user with userID: {user_id}:")
# for index, row in df_rec2.iterrows():
#     print(f"title: {row['Book-Title']} & url: {row['Image-URL-L']}")
#     print("-" * 25)
