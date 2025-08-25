import pickle
import numpy as np
from pathlib import Path

def simulate_rating_impact(movie_title, movie_title_to_id, ratings=[1.0, 3.0, 5.0], top_k=3):
    """
    Given a movie title and potential ratings (1, 3, 5), simulate the top-k recommendations
    the user would get under each rating scenario.
    """
    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / "mf_model.pkl"
    movies_path = BASE_DIR / "data" / "movies.csv"

    # Load model and movie data
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    V = model["V"]
    movie_id_map = model["movie_id_map"]
    reverse_movie_id_map = model["reverse_movie_id_map"]

    import pandas as pd
    movies_df = pd.read_csv(movies_path)
    movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

    # Get movie index in V
    movie_id = movie_title_to_id.get(movie_title)
    if movie_id is None or movie_id not in movie_id_map:
        return {}

    movie_idx = movie_id_map[movie_id]
    K = V.shape[1]
    lambda_reg = 0.1
    learning_rate = 0.01

    rating_predictions = {}

    for rating in ratings:
        U_new = np.random.normal(scale=1./K, size=K)
        for _ in range(30):  # SGD steps
            err = rating - np.dot(U_new, V[movie_idx])
            U_new += learning_rate * (err * V[movie_idx] - lambda_reg * U_new)

        preds = U_new @ V.T
        top_indices = np.argsort(-preds)[:top_k]
        top_ids = [reverse_movie_id_map[i] for i in top_indices]
        top_titles = [movie_id_to_title[mid] for mid in top_ids if mid in movie_id_to_title]

        rating_predictions[str(int(rating)) + "_stars"] = top_titles

    return rating_predictions
