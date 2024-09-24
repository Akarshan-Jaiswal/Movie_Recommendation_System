import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

# Load the MovieLens dataset
ratings = tfds.load('movielens/100k-ratings', split="train")
movies = tfds.load('movielens/100k-movies', split="train")

# Preprocess the dataset
ratings = ratings.map(lambda x: {
    'movie_id': x['movie_id'],
    'user_id': x['user_id'],
    'user_rating': x['user_rating']
})

# Get unique user and movie IDs
movie_ids = ratings.map(lambda x: x['movie_id']).unique()
user_ids = ratings.map(lambda x: x['user_id']).unique()


# Create embedding model
class MovieLensModel(tf.keras.Model):
    def __init__(self, num_users, num_movies, embedding_dim):
        super().__init__()
        # Embedding layers
        self.user_embedding = tf.keras.layers.Embedding(num_users, embedding_dim)
        self.movie_embedding = tf.keras.layers.Embedding(num_movies, embedding_dim)
        # Compute dot product between user and movie embeddings
        self.dot = tf.keras.layers.Dot(axes=1)

    def call(self, inputs):
        user_id, movie_id = inputs
        user_vec = self.user_embedding(user_id)
        movie_vec = self.movie_embedding(movie_id)
        return self.dot([user_vec, movie_vec])


# Create a model instance
num_users = 1000  # Example: max unique users
num_movies = 1000  # Example: max unique movies
embedding_dim = 50  # Size of the embedding vectors

model = MovieLensModel(num_users, num_movies, embedding_dim)

# Compile the model
model.compile(optimizer='adam', loss='mse')


# Prepare training data (user_ids, movie_ids, ratings)
def prepare_data(ratings):
    user_ids = []
    movie_ids = []
    ratings_ = []

    for rating in ratings:
        user_ids.append(rating['user_id'].numpy())
        movie_ids.append(rating['movie_id'].numpy())
        ratings_.append(rating['user_rating'].numpy())

    return np.array(user_ids), np.array(movie_ids), np.array(ratings_)


# Split data into training and validation sets
train_ratings = ratings.take(80000)
test_ratings = ratings.skip(80000)

train_user_ids, train_movie_ids, train_ratings = prepare_data(train_ratings)
test_user_ids, test_movie_ids, test_ratings = prepare_data(test_ratings)

# Train the model
model.fit([train_user_ids, train_movie_ids], train_ratings, epochs=5,
          validation_data=([test_user_ids, test_movie_ids], test_ratings))

# Predict ratings for new user-movie pairs
predicted_ratings = model.predict([test_user_ids, test_movie_ids])


# Recommend top movies for a user
def recommend_movies(user_id, num_recommendations=10):
    # Predict ratings for all movies for a specific user
    movie_ids = np.arange(num_movies)
    predicted_ratings = model.predict([np.full(movie_ids.shape, user_id), movie_ids])

    # Get top movie IDs
    top_movie_ids = np.argsort(predicted_ratings, axis=0)[-num_recommendations:]
    return top_movie_ids


# Example: Recommend top 10 movies for user 42
recommendations = recommend_movies(42, 10)
print(f"Recommended movie IDs for user 42: {recommendations}")
