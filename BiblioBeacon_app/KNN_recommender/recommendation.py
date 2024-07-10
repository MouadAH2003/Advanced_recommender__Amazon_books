import pandas as pd
import joblib

# Load the best model and the dataframe
best_model = joblib.load('./KNN_recommender/knn_model.pkl')
df = pd.read_pickle('./KNN_recommender/df.pkl')

def KNNRecommender_GridSearch(title="", top_n=6):
    if title not in df.index:
        print(f"The given book '{title}' does not exist in the dataset.")
        return None
    
    distances, indices = best_model.kneighbors([df.loc[title].values], n_neighbors=top_n)
    recommended_books = pd.DataFrame({
        "title": df.iloc[indices[0]].index.values,
        "distance": distances[0]
    }).sort_values(by="distance", ascending=True).head(top_n)["title"].values
    
    return recommended_books[1:].tolist()
