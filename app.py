from fastapi import FastAPI
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = FastAPI()

# Example movie feature matrix
movie_features = np.array([
    [8.5, 90, 1, 0, 0],  # Movie 1
    [7.2, 75, 0, 1, 0],
    [8.0, 85, 0, 0, 1]
])

knn = NearestNeighbors(n_neighbors=3, metric='cosine')
knn.fit(movie_features)

@app.get("/recommend/{movie_index}")
def recommend(movie_index: int):
    distances, indices = knn.kneighbors(movie_features[movie_index].reshape(1, -1), n_neighbors=3)
    return {"recommended_movie_indices": indices.tolist()}
