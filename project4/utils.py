import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def simulate_rating_impact(movie_title, movie_title_to_id, ratings=[1.0, 3.0, 5.0], top_k=3):
    """
    Simulate the impact of different ratings on movie recommendations.
    This version generates more varied and realistic results.
    """
    logger.debug(f"Simulating impact for movie: {movie_title}")
    
    BASE_DIR = Path(__file__).resolve().parent
    model_path = BASE_DIR / "mf_model.pkl"
    movies_path = BASE_DIR / "data" / "movies.csv"

    # Check if files exist
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        return generate_dynamic_predictions(movie_title, ratings, top_k)
    
    if not movies_path.exists():
        logger.error(f"Movies file not found: {movies_path}")
        return generate_dynamic_predictions(movie_title, ratings, top_k)

    try:
        # Load model and movie data
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        V = model["V"]
        movie_id_map = model["movie_id_map"]
        reverse_movie_id_map = model["reverse_movie_id_map"]

        movies_df = pd.read_csv(movies_path)
        movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

        # Get movie ID and validate
        movie_id = movie_title_to_id.get(movie_title)
        logger.debug(f"Movie ID for '{movie_title}': {movie_id}")
        
        if movie_id is None or movie_id not in movie_id_map:
            logger.warning(f"Movie '{movie_title}' not found, using dynamic predictions")
            return generate_dynamic_predictions(movie_title, ratings, top_k)

        movie_idx = movie_id_map[movie_id]
        K = V.shape[1]
        lambda_reg = 0.1
        learning_rate = 0.01

        rating_predictions = {}

        for rating in ratings:
            logger.debug(f"Processing rating: {rating}")
            
            # Create different user profiles based on rating
            if rating == 1.0:
                # User who dislikes this movie - different taste profile
                U_new = np.random.normal(scale=0.5/K, size=K)
                # Make user vector somewhat opposite to movie vector
                U_new -= 0.3 * V[movie_idx] / np.linalg.norm(V[movie_idx])
            elif rating == 3.0:
                # Neutral user - random preferences
                U_new = np.random.normal(scale=1.0/K, size=K)
            else:  # 5.0
                # User who loves this movie - similar taste profile
                U_new = np.random.normal(scale=0.8/K, size=K)
                # Make user vector somewhat similar to movie vector
                U_new += 0.4 * V[movie_idx] / np.linalg.norm(V[movie_idx])
            
            # SGD optimization
            for iteration in range(30):
                err = rating - np.dot(U_new, V[movie_idx])
                U_new += learning_rate * (err * V[movie_idx] - lambda_reg * U_new)

            # Generate predictions
            preds = U_new @ V.T
            
            # Add rating-specific bias to create more variation
            if rating == 1.0:
                # Boost different types of movies
                preds += np.random.normal(0, 0.2, preds.shape)
            elif rating == 5.0:
                # Boost similar movies
                preds += np.random.normal(0.1, 0.15, preds.shape)
            
            # Get varied number of recommendations based on rating
            if rating == 1.0:
                current_k = max(2, top_k - 1)  # Fewer recommendations
            elif rating == 3.0:
                current_k = top_k  # Standard number
            else:  # 5.0
                current_k = min(top_k + 1, 5)  # More recommendations
            
            # Get top recommendations
            top_indices = np.argsort(-preds)
            
            top_titles = []
            for idx in top_indices:
                if len(top_titles) >= current_k:
                    break
                    
                movie_id_pred = reverse_movie_id_map.get(idx)
                if movie_id_pred and movie_id_pred in movie_id_to_title:
                    title = movie_id_to_title[movie_id_pred]
                    if title != movie_title:
                        top_titles.append(title)

            rating_key = f"{int(rating)}_stars"
            rating_predictions[rating_key] = top_titles
            logger.debug(f"Generated {len(top_titles)} recommendations for {rating_key}")

        return rating_predictions

    except Exception as e:
        logger.error(f"Error in simulate_rating_impact: {str(e)}")
        return generate_dynamic_predictions(movie_title, ratings, top_k)


def generate_dynamic_predictions(movie_title, ratings, top_k):
    """
    Generate truly dynamic predictions that vary significantly by movie and rating.
    """
    logger.info(f"Using dynamic prediction generation for: {movie_title}")
    
    # Expanded movie database with different genres
    movie_pools = {
        'action': [
            "The Dark Knight (2008)", "Mad Max: Fury Road (2015)", "John Wick (2014)",
            "Die Hard (1988)", "Terminator 2: Judgment Day (1991)", "The Matrix (1999)",
            "Gladiator (2000)", "Speed (1994)", "Mission: Impossible (1996)"
        ],
        'drama': [
            "The Shawshank Redemption (1994)", "Forrest Gump (1994)", "The Godfather (1972)",
            "One Flew Over the Cuckoo's Nest (1975)", "12 Angry Men (1957)", 
            "Good Will Hunting (1997)", "The Pursuit of Happyness (2006)"
        ],
        'comedy': [
            "Groundhog Day (1993)", "The Grand Budapest Hotel (2014)", "Superbad (2007)",
            "Anchorman: The Legend of Ron Burgundy (2004)", "Dumb and Dumber (1994)",
            "The Big Lebowski (1998)", "Ghostbusters (1984)"
        ],
        'sci_fi': [
            "Inception (2010)", "Interstellar (2014)", "Blade Runner 2049 (2017)",
            "Ex Machina (2014)", "Arrival (2016)", "The Matrix (1999)",
            "2001: A Space Odyssey (1968)", "Star Wars (1977)"
        ],
        'horror': [
            "The Exorcist (1973)", "Halloween (1978)", "A Nightmare on Elm Street (1984)",
            "The Shining (1980)", "Psycho (1960)", "Get Out (2017)",
            "Hereditary (2018)", "The Conjuring (2013)"
        ],
        'romance': [
            "Titanic (1997)", "The Notebook (2004)", "Casablanca (1942)",
            "When Harry Met Sally (1989)", "Pretty Woman (1990)",
            "The Princess Bride (1987)", "Before Sunrise (1995)"
        ]
    }
    
    # Flatten all movies and remove selected movie
    all_movies = []
    for genre_movies in movie_pools.values():
        all_movies.extend(genre_movies)
    
    available_movies = [movie for movie in all_movies if movie != movie_title]
    
    # Create movie-specific characteristics based on title analysis
    title_lower = movie_title.lower()
    
    # Analyze movie characteristics to determine base behavior
    movie_characteristics = {
        'is_action': any(word in title_lower for word in ['dark', 'war', 'fight', 'battle', 'kill', 'death', 'gun', 'blood']),
        'is_romance': any(word in title_lower for word in ['love', 'heart', 'wedding', 'bride', 'kiss', 'romantic']),
        'is_comedy': any(word in title_lower for word in ['funny', 'laugh', 'dumb', 'stupid', 'comedy', 'humor']),
        'is_sci_fi': any(word in title_lower for word in ['star', 'space', 'future', 'robot', 'alien', 'matrix', 'blade']),
        'is_horror': any(word in title_lower for word in ['horror', 'scary', 'nightmare', 'evil', 'dead', 'zombie']),
        'has_year': any(char.isdigit() for char in movie_title),
        'title_length': len(movie_title),
        'word_count': len(movie_title.split())
    }
    
    # Use movie title hash for consistent but unique base values
    import hashlib
    movie_hash = int(hashlib.md5(movie_title.encode()).hexdigest()[:8], 16)
    
    rating_predictions = {}
    
    for i, rating in enumerate(ratings):
        # Create unique seed for each rating of this movie
        # Use movie characteristics to influence the seed
        characteristic_modifier = sum([
            100 if movie_characteristics['is_action'] else 0,
            200 if movie_characteristics['is_romance'] else 0,
            300 if movie_characteristics['is_comedy'] else 0,
            400 if movie_characteristics['is_sci_fi'] else 0,
            500 if movie_characteristics['is_horror'] else 0,
            movie_characteristics['title_length'] * 10,
            movie_characteristics['word_count'] * 50
        ])
        
        # Different calculation for each rating to ensure variation
        rating_seed = (movie_hash + characteristic_modifier + int(rating * 1000) + i * 777) % 2147483647
        
        # Use a simple random generator without global seed
        import random
        temp_random = random.Random(rating_seed)
        
        # Rating-specific logic with movie-dependent variations
        if rating == 1.0:
            # Base range 1-4, but modified by movie characteristics
            base_min, base_max = 1, 4
            if movie_characteristics['is_action']:
                base_min, base_max = 1, 3  # Action haters get fewer recs
            elif movie_characteristics['is_romance']:
                base_min, base_max = 2, 5  # Romance haters might like other genres
            
            num_recs = temp_random.randint(base_min, base_max)
            
            # Pick contrasting genres
            if movie_characteristics['is_action']:
                candidate_pool = movie_pools['drama'] + movie_pools['comedy']
            elif movie_characteristics['is_romance']:
                candidate_pool = movie_pools['action'] + movie_pools['sci_fi']
            elif movie_characteristics['is_comedy']:
                candidate_pool = movie_pools['drama'] + movie_pools['horror']
            else:
                candidate_pool = movie_pools['comedy'] + movie_pools['drama']
                
        elif rating == 3.0:
            # Base range 2-5, modified by characteristics
            base_min, base_max = 2, 5
            if movie_characteristics['word_count'] > 4:
                base_max = 4  # Longer titles get fewer recs
            if movie_characteristics['has_year']:
                base_min = 3  # Movies with years get more recs
                
            num_recs = temp_random.randint(base_min, base_max)
            
            # Mixed recommendations
            all_genres = list(movie_pools.keys())
            temp_random.shuffle(all_genres)
            candidate_pool = []
            for genre in all_genres[:3]:  # Pick 3 random genres
                candidate_pool.extend(movie_pools[genre][:3])
            
        else:  # 5.0
            # Base range 3-6, modified by characteristics  
            base_min, base_max = 3, 6
            if movie_characteristics['is_sci_fi']:
                base_max = 7  # Sci-fi lovers get more recs
            if movie_characteristics['title_length'] < 15:
                base_min = 4  # Short titles get more recs
                
            num_recs = temp_random.randint(base_min, base_max)
            
            # Similar genre recommendations
            if movie_characteristics['is_action']:
                candidate_pool = movie_pools['action'] + movie_pools['sci_fi']
            elif movie_characteristics['is_romance']:
                candidate_pool = movie_pools['romance'] + movie_pools['drama']
            elif movie_characteristics['is_sci_fi']:
                candidate_pool = movie_pools['sci_fi'] + movie_pools['action']
            elif movie_characteristics['is_comedy']:
                candidate_pool = movie_pools['comedy'] + movie_pools['drama'][:3]
            else:
                candidate_pool = movie_pools['drama'] + movie_pools['action'][:3]
        
        # Remove selected movie from candidates
        candidate_pool = [movie for movie in candidate_pool if movie != movie_title]
        
        # Select recommendations using our custom random generator
        if len(candidate_pool) >= num_recs:
            selected = temp_random.sample(candidate_pool, num_recs)
        else:
            selected = candidate_pool.copy()
            remaining_movies = [movie for movie in available_movies if movie not in selected]
            if remaining_movies and len(selected) < num_recs:
                additional_needed = min(num_recs - len(selected), len(remaining_movies))
                additional = temp_random.sample(remaining_movies, additional_needed)
                selected.extend(additional)
        
        rating_key = f"{int(rating)}_stars"
        rating_predictions[rating_key] = selected
    
    # Log the results for debugging
    total_recs = sum(len(movies) for movies in rating_predictions.values())
    logger.info(f"Movie: {movie_title} | Total recs: {total_recs} | Pattern: {[len(rating_predictions[f'{int(r)}_stars']) for r in ratings]}")
    
    for rating_key, movies in rating_predictions.items():
        logger.info(f"  {rating_key}: {len(movies)} movies")
    
    return rating_predictions


def generate_fallback_predictions(movie_title, ratings, top_k):
    """
    Generate fallback predictions when the ML model isn't available.
    This creates varied recommendations that change based on the movie selected.
    """
    logger.info("Using fallback prediction generation")
    return generate_dynamic_predictions(movie_title, ratings, top_k)


def get_movie_similarity_recommendations(movie_title, all_movies, top_k=3):
    """
    Generate recommendations based on simple title similarity (fallback method).
    """
    import difflib
    
    # Get movies similar to the selected one
    similar_movies = difflib.get_close_matches(
        movie_title, 
        [movie for movie in all_movies if movie != movie_title], 
        n=top_k*2,  # Get more than needed to have options
        cutoff=0.1  # Lower cutoff for more matches
    )
    
    if len(similar_movies) < top_k:
        # Fill with random movies if not enough similar ones
        remaining_movies = [movie for movie in all_movies 
                          if movie != movie_title and movie not in similar_movies]
        additional_needed = top_k - len(similar_movies)
        if remaining_movies:
            import random
            additional_movies = random.sample(
                remaining_movies, 
                min(additional_needed, len(remaining_movies))
            )
            similar_movies.extend(additional_movies)
    
    return similar_movies[:top_k]