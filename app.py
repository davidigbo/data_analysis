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


# import streamlit as st
# import requests

# st.title("Movie Recommendation Engine")

# movie_index = st.number_input("Enter a movie index:", min_value=0, value=0)
# if st.button("Get Recommendations"):
#     response = requests.get(f"http://127.0.0.1:8000/recommend/{movie_index}")
#     recommendations = response.json().get("recommended_movie_indices", [])
#     st.write("Recommended Movie Indices:", recommendations)

import requests

BACKEND_URL = "https://data-analysis-2-pfv6.onrender.com/"

movie_index = 2  # Example movie index
response = requests.get(f"{BACKEND_URL}/recommend/{movie_index}")

if response.status_code == 200:
    recommendations = response.json()
    st.write("Recommended movies:", recommendations)
else:
    st.write("Error fetching recommendations")
