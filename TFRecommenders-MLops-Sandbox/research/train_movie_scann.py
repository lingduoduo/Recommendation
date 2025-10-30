# https://github.meetmecorp.com/lhuang/deep-recommender-sandbox/blob/master/notebooks/Recommenders/8-Efficient_serving_20220509.ipynb

#!/usr/bin/env python
# coding: utf-8


import os
import tempfile
from typing import Dict, Text

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# In[5]:


# And load the data:

# In[6]:


# Load the MovieLens 100K data.
ratings = tfds.load("movielens/100k-ratings", split="train")

# Get the ratings data.
ratings = (
    ratings
    # Retain only the fields we need.
    .map(lambda x: {"user_id": x["user_id"], "movie_title": x["movie_title"]})
    # Cache for efficiency.
    .cache(tempfile.NamedTemporaryFile().name)
)

# Get the movies data.
movies = tfds.load("movielens/100k-movies", split="train")
movies = (
    movies
    # Retain only the fields we need.
    .map(lambda x: x["movie_title"])
    # Cache for efficiency.
    .cache(tempfile.NamedTemporaryFile().name)
)

# Before we can build a model, we need to set up the user and movie vocabularies:

# In[7]:


user_ids = ratings.map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(user_ids.batch(1000))))

# We'll also set up the training and test sets:

# In[8]:


tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)


# ### Model definition
#
# Just as in the [basic retrieval](https://www.tensorflow.org/recommenders/examples/basic_retrieval) tutorial, we build a simple two-tower model.

# In[9]:


class MovielensModel(tfrs.Model):
    def __init__(self):
        super().__init__()

        embedding_dimension = 32

        # Set up a model for representing movies.
        self.movie_model = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_movie_titles, mask_token=None
                ),
                # We add an additional embedding to account for unknown tokens.
                tf.keras.layers.Embedding(
                    len(unique_movie_titles) + 1, embedding_dimension
                ),
            ]
        )

        # Set up a model for representing users.
        self.user_model = tf.keras.Sequential(
            [
                tf.keras.layers.StringLookup(
                    vocabulary=unique_user_ids, mask_token=None
                ),
                # We add an additional embedding to account for unknown tokens.
                tf.keras.layers.Embedding(
                    len(unique_user_ids) + 1, embedding_dimension
                ),
            ]
        )

        # Set up a task to optimize the model and compute metrics.
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).cache().map(self.movie_model)
            )
        )

    def compute_loss(
        self, features: Dict[Text, tf.Tensor], training=False
    ) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics.

        return self.task(
            user_embeddings, positive_movie_embeddings, compute_metrics=not training
        )


# ### Fitting and evaluation
#
# A TFRS model is just a Keras model. We can compile it:

# In[10]:


model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# Estimate it:

# In[11]:


model.fit(train.batch(8192), epochs=1)

# And evaluate it.

# In[12]:


model.evaluate(test.batch(8192), return_dict=True)

# ## Approximate prediction
#
# The most straightforward way of retrieving top candidates in response to a query is to do it via brute force: compute user-movie scores for all possible movies, sort them, and pick a couple of top recommendations.
#
# In TFRS, this is accomplished via the `BruteForce` layer:

# In[13]:


brute_force = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
brute_force.index_from_dataset(
    movies.batch(128).map(lambda title: (title, model.movie_model(title)))
)

# Get predictions for user 42.
_, titles = brute_force(np.array(["42"]), k=3)
print(f"Top recommendations: {titles[0]}")

# Using ScaNN in TFRS is accomplished via the `tfrs.layers.factorized_top_k.ScaNN` layer. It follow the same interface as the other top k layers:
scann = tfrs.layers.factorized_top_k.ScaNN(num_reordering_candidates=100)
scann.index_from_dataset(
    movies.batch(128).map(lambda title: (title, model.movie_model(title)))
)

_, titles = scann(model.user_model(np.array(["42"])), k=3)
print(f"Top recommendations: {titles[0]}")

# ## Evaluating the approximation
# # Construct a dataset of movies that's 1,000 times larger. We
# # do this by adding several million dummy movie titles to the dataset.
# lots_of_movies = tf.data.Dataset.concatenate(
# 	movies.batch(4096),
# 	movies.batch(4096).repeat(1_000).map(lambda x: tf.zeros_like(x))
# )
#
# # We also add lots of dummy embeddings by randomly perturbing
# # the estimated embeddings for real movies.
# lots_of_movies_embeddings = tf.data.Dataset.concatenate(
# 	movies.batch(4096).map(model.movie_model),
# 	movies.batch(4096).repeat(1_000)
# 		.map(lambda x: model.movie_model(x))
# 		.map(lambda x: x * tf.random.uniform(tf.shape(x)))
# )
#
# # Override the existing streaming candidate source.
# model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(
# 	candidates = lots_of_movies_embeddings
# )
# # Need to recompile the model for the changes to take effect.
# model.compile()
# start = time.time()
# baseline_result = model.evaluate(test.batch(8192), return_dict = True, verbose = False)
# end = time.time()
# print("Latency (ms):", 1000 * (end - start))
# # Latency (ms): 595446.6128349304
#
# # ScaNN model
# model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(
# 	candidates = scann
# )
# model.compile()
#
# # We can use a much bigger batch size here because ScaNN evaluation
# # is more memory efficient.
# start = time.time()
# scann_result = model.evaluate(test.batch(8192), return_dict = True, verbose = False)
# end = time.time()
# print("Latency (ms):", 1000 * (end - start))
# # Latency (ms): 3137.045383453369


# Deploy the approximate model
scann_index = tfrs.layers.factorized_top_k.ScaNN(model.user_model)
scann_index.index_from_dataset(
    tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)

# Get recommendations.
_, titles = scann_index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "model")

    # Save the index.
    tf.saved_model.save(
        scann_index,
        path,
        options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"]),
    )

    # Load it back; can also be done in TensorFlow Serving.
    loaded = tf.saved_model.load(path)

    # Pass a user id in, get top predicted movie titles back.
    scores, titles = loaded(["42"])

    print(f"Recommendations: {titles[0][:3]}")
