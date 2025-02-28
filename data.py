from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd

movie_features = np.array([
    [8.5, 90, 1, 0, 0],
    [8.1, 120, 0, 1, 0],
    [7.3, 95, 0, 0, 1],
    [8.5, 100, 1, 0, 0],
    [8.1, 130, 0, 1, 0],
    [7.3, 100, 0, 0, 1]
])

knn = NearestNeighbors(n_neighbors=5, metric='cosine').fit(movie_features)

distances, indices = knn.kneighbors(movie_features[0].reshape(1, -1), n_neighbors=3)

print("Recommendations for movie indices: ", indices)
print("Recommendations for movie distances: ", distances)

from sklearn.metrics import precision_score, recall_score

# Example: 1 = liked, 0 = disliked
y_true = [1, 0, 1, 1, 0, 1]  # Actual user preferences
y_pred = [1, 1, 1, 0, 0, 1]  # Model recommendations

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")

control_group = np.array([20, 18, 22, 25, 19])  # Minutes spent without recommendations
test_group = np.array([5, 7, 6, 4, 8])  # Minutes spent with recommendations

avg_time_saved = np.mean(control_group) - np.mean(test_group)
print(f"Average Time Saved: {avg_time_saved:.2f} minutes")
