import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def train_matrix_factorization():
    """
    Train matrix factorization model on MovieLens data and save it
    This creates the mf_model.pkl file that your utils.py uses
    """
    print("Starting Matrix Factorization Training...")
    
    # Load data
    BASE_DIR = Path(__file__).resolve().parent
    ratings_path = BASE_DIR / "data" / "ratings.csv"
    movies_path = BASE_DIR / "data" / "movies.csv"
    
    # Check if files exist
    if not ratings_path.exists():
        print(f"ERROR: {ratings_path} not found!")
        return None
    if not movies_path.exists():
        print(f"ERROR: {movies_path} not found!")
        return None
    
    print("Loading data...")
    ratings_df = pd.read_csv(ratings_path)
    movies_df = pd.read_csv(movies_path)
    
    print(f"Loaded {len(ratings_df)} ratings and {len(movies_df)} movies")
    
    # Create user and movie mappings
    unique_users = ratings_df['userId'].unique()
    unique_movies = ratings_df['movieId'].unique()
    
    print(f"Found {len(unique_users)} unique users and {len(unique_movies)} unique movies")
    
    user_id_map = {user_id: idx for idx, user_id in enumerate(unique_users)}
    movie_id_map = {movie_id: idx for idx, movie_id in enumerate(unique_movies)}
    reverse_movie_id_map = {idx: movie_id for movie_id, idx in movie_id_map.items()}
    
    # Create rating matrix dimensions
    n_users = len(unique_users)
    n_movies = len(unique_movies)
    
    print(f"Matrix dimensions: {n_users} users x {n_movies} movies")
    
    # Matrix Factorization parameters
    K = 50  # latent factors
    lambda_reg = 0.1
    learning_rate = 0.01
    n_epochs = 100
    
    print(f"Training parameters: K={K}, lambda={lambda_reg}, lr={learning_rate}, epochs={n_epochs}")
    
    # Initialize matrices randomly
    np.random.seed(42)  # For reproducible results
    U = np.random.normal(scale=1./K, size=(n_users, K))
    V = np.random.normal(scale=1./K, size=(n_movies, K))
    
    print("Initialized U and V matrices")
    
    # Convert ratings to matrix indices for faster training
    print("Preparing training data...")
    ratings_matrix = []
    for _, row in ratings_df.iterrows():
        user_idx = user_id_map[row['userId']]
        movie_idx = movie_id_map[row['movieId']]
        rating = row['rating']
        ratings_matrix.append((user_idx, movie_idx, rating))
    
    print(f"Prepared {len(ratings_matrix)} training samples")
    print("Starting SGD training...")
    
    # SGD training loop
    for epoch in range(n_epochs):
        total_error = 0
        np.random.shuffle(ratings_matrix)  # Shuffle for better convergence
        
        # Process each rating
        for user_idx, movie_idx, rating in ratings_matrix:
            # Predict current rating
            prediction = np.dot(U[user_idx], V[movie_idx])
            error = rating - prediction
            total_error += error ** 2
            
            # Update user and movie factors using gradient descent
            U_user_old = U[user_idx].copy()
            U[user_idx] += learning_rate * (error * V[movie_idx] - lambda_reg * U[user_idx])
            V[movie_idx] += learning_rate * (error * U_user_old - lambda_reg * V[movie_idx])
        
        # Calculate RMSE for this epoch
        rmse = np.sqrt(total_error / len(ratings_matrix))
        
        # Print progress every 20 epochs
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}, RMSE: {rmse:.4f}")
    
    final_rmse = np.sqrt(total_error / len(ratings_matrix))
    print(f"Training completed! Final RMSE: {final_rmse:.4f}")
    
    # Create model dictionary with all necessary components
    model = {
        'U': U,  # User factors
        'V': V,  # Movie factors  
        'user_id_map': user_id_map,  # Maps original user IDs to matrix indices
        'movie_id_map': movie_id_map,  # Maps original movie IDs to matrix indices
        'reverse_movie_id_map': reverse_movie_id_map,  # Maps matrix indices back to movie IDs
        'K': K,  # Number of latent factors
        'lambda_reg': lambda_reg,  # Regularization parameter
        'final_rmse': final_rmse  # Training performance
    }
    
    # Save the trained model
    model_path = BASE_DIR / "mf_model.pkl"
    print(f"Saving model to {model_path}...")
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print("✅ Model saved successfully!")
    print(f"📁 Model file: {model_path}")
    print(f"📊 Model contains {n_users} users and {n_movies} movies")
    print(f"🎯 Final RMSE: {final_rmse:.4f}")
    
    return model

if __name__ == "__main__":
    print("=" * 60)
    print("🎬 MOVIELENS MATRIX FACTORIZATION TRAINING")
    print("=" * 60)
    
    model = train_matrix_factorization()
    
    if model is not None:
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETED SUCCESSFULLY!")
        print("You can now run your Django application.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ TRAINING FAILED!")
        print("Please check that your data files are in the correct location.")
        print("=" * 60)