#!/usr/bin/env python
# coding: utf-8

# ### Setup envs

# In[1]:
import pprint
import logging
import time

import numpy as np
import pandas as pd
import re

import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_recommenders as tfrs


from typing import Dict, Text
import tensorflow_recommenders as tfrs
import tensorflow as tf
import pandas as pd
import numpy as np
from sagemaker import get_execution_role

role = get_execution_role()
bucket = "ling-cold-start-data"
prefix = "2021-11-22"
data_key = "2021-11-22.csv"
data_location = "s3://{}/{}/{}".format(bucket, prefix, data_key)
print(data_location)

# In[5]:


# Model definition


# In[6]:
class UserModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()

        self.gender_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=conf["unique_genders"], mask_token=None
                ),
                tf.keras.layers.Embedding(len(conf["unique_genders"]) + 1, 4),
            ]
        )

        self.lang_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=conf["unique_langs"], mask_token=None
                ),
                tf.keras.layers.Embedding(len(conf["unique_langs"]) + 1, 10),
            ]
        )

        self.country_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=conf["unique_countries"], mask_token=None
                ),
                tf.keras.layers.Embedding(len(conf["unique_countries"]) + 1, 10),
            ]
        )

        self.network_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=conf["unique_networks"], mask_token=None
                ),
                tf.keras.layers.Embedding(len(conf["unique_networks"]) + 1, 4),
            ]
        )

        age_boundaries = np.array(
            [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, float("inf")]
        )
        self.viewer_age_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.Discretization(
                    age_boundaries.tolist()
                ),
                tf.keras.layers.Embedding(len(age_boundaries), 2),
            ]
        )

        self.centroids = tf.constant(
            [
                [36.68147669256268, -82.8910274009993],
                [23.22243322909555, 78.23027450833709],
                [50.04997682638993, 0.22379313938744885],
                [37.9309447099281, -117.00741350764692],
                [-32.795864819917725, 148.7159172660312],
                [-18.570548393114084, -54.280255665692565],
                [13.921140442819565, 116.38740315555172],
                [29.78951080730802, 40.279515865947936],
            ]
        )
        self.viewer_lat_long_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.TextVectorization(
                    standardize=None,
                    split=self.classify,
                    vocabulary=[str(i) for i in range(len(self.centroids))],
                    max_tokens=len(self.centroids) + 2,
                ),
                tf.keras.layers.Embedding(len(self.centroids) + 2, 2),
            ]
        )

    @tf.function()
    def call(self, inputs):
        return tf.concat(
            [
                self.gender_embedding(inputs["viewer_gender"]),
                self.lang_embedding(inputs["viewer_lang"]),
                self.country_embedding(inputs["viewer_country"]),
                self.network_embedding(inputs["viewer_network"]),
                self.viewer_age_embedding(inputs["viewer_age"]),
                self.viewer_lat_long_embedding(inputs["viewer_lat_long"]),
            ],
            axis=1,
        )

    @tf.keras.utils.register_keras_serializable()
    def classify(self, pair):
        """
        given a datapoint, compute the cluster closest to the datapoint. Return the cluster ID of that cluster.
        :param pair:
        :return: cluster ID
        """
        str_data = tf.strings.split(pair, sep=",").values
        str_data = tf.map_fn(lambda x: tf.strings.regex_replace(x, "b'", ""), str_data)
        datapoints = tf.map_fn(
            lambda x: tf.strings.to_number(x), str_data, dtype=(tf.float32)
        )
        datapoints = tf.reshape(datapoints, [-1, 2])

        expanded_centroids = tf.expand_dims(self.centroids, 1)
        expanded_vectors = tf.expand_dims(datapoints, 0)
        distances = tf.reduce_sum(
            tf.square(tf.subtract(expanded_vectors, expanded_centroids)), 2
        )
        clusters = tf.math.argmin(distances)
        return tf.strings.as_string(clusters)


# In[7]:


class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, conf):
        """Model for encoding user queries.

        Args:
          layer_sizes:
                A list of integers where the i-th entry represents the number of units
                the i-th layer contains.
        """
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = UserModel(conf)
        self.dense_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    32,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(0.0001),
                ),
                tf.keras.layers.Dense(32),
            ]
        )

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


# In[8]:


class BroadcasterModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()

        self.broadcaster_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=conf["unique_broadcasters"], mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(conf["unique_broadcasters"]) + 1,
                    conf["broadcaster_embedding_dimension"],
                ),
            ]
        )

    def call(self, broadcaster):
        return tf.concat(
            [
                self.broadcaster_embedding(broadcaster),
            ],
            axis=1,
        )


# In[9]:


class CandidateModel(tf.keras.Model):
    """Model for encoding movies."""

    def __init__(self, conf):
        """Model for encoding movies.

        Args:
          layer_sizes:
                A list of integers where the i-th entry represents the number of units
                the i-th layer contains.
        """
        super().__init__()

        self.embedding_model = BroadcasterModel(conf)

        self.dense_layers = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    32,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(0.0001),
                ),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(32),
            ]
        )

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class RankingModel(tf.keras.Model):
    def __init__(self, conf):
        super().__init__()
        self.embedding_dimension = conf["broadcaster_embedding_dimension"]

        # Compute predictions.
        self.ratings = tf.keras.Sequential(
            [
                # Learn multiple dense layers.
                tf.keras.layers.Dense(self.embedding_dimension * 4, activation="relu"),
                tf.keras.layers.Dense(self.embedding_dimension, activation="relu"),
                # Make rating predictions in the final layer.
                tf.keras.layers.Dense(1),
            ]
        )

    def call(self, inputs):
        query_embeddings, positive_broadcaster_embeddings = inputs
        return self.ratings(
            tf.concat([query_embeddings, positive_broadcaster_embeddings], axis=1)
        )


# Load data

# In[10]:


def load_data_file_cold(file, stats):
    print("loading file:" + file)
    training_df = pd.read_csv(
        file,
        skiprows=[0],
        names=[
            "viewer",
            "broadcaster",
            "viewer_age",
            "viewer_gender",
            "viewer_longitude",
            "viewer_latitude",
            "viewer_lang",
            "viewer_country",
            "broadcaster_age",
            "broadcaster_gender",
            "broadcaster_longitude",
            "broadcaster_latitude",
            "broadcaster_lang",
            "broadcaster_country",
            "duration",
            "viewer_network",
            "broadcaster_network",
            "viewer_lat_long_cluster",
            "rank",
        ],
        dtype={
            "viewer": np.unicode,
            "broadcaster": np.unicode,
            "viewer_age": np.single,
            "viewer_gender": np.unicode,
            "viewer_longitude": np.single,
            "viewer_latitude": np.single,
            "viewer_lang": np.unicode,
            "viewer_country": np.unicode,
            "broadcaster_age": np.single,
            "broadcaster_longitude": np.single,
            "broadcaster_latitude": np.single,
            "broadcaster_lang": np.unicode,
            "broadcaster_country": np.unicode,
            "viewer_network": np.unicode,
            "broadcaster_network": np.unicode,
            "viewer_lat_long_cluster": np.unicode,
            "rank": np.unicode,
        },
    )

    values = {
        "viewer": "unknown",
        "broadcaster": "unknown",
        "viewer_age": 30,
        "viewer_gender": "unknown",
        "viewer_longitude": 0,
        "viewer_latitude": 0,
        "viewer_lang": "unknown",
        "viewer_country": "unknown",
        "broadcaster_age": 30,
        "broadcaster_longitude": 0,
        "broadcaster_latitude": 0,
        "broadcaster_lang": "unknown",
        "broadcaster_country": "unknown",
        "duration": 0,
        "viewer_network": "unknown",
        "broadcaster_network": "unknown",
        "viewer_lat_long": tf.constant(["40.36393,-74.89611"]),
        "rank": "1",
    }
    training_df = training_df.sample(n=10000)
    training_df.fillna(value=values, inplace=True)
    training_df["viewer_lat_long"] = training_df[
        ["viewer_latitude", "viewer_longitude"]
    ].apply(lambda x: "{},{}".format(x[0], x[1]), axis=1)
    print(training_df.head(10))
    print(training_df.iloc[-10:])
    # stats.send_stats('data-size', len(training_df.index))
    return training_df


def load_training_data_cold(file, stats):
    ratings_df = load_data_file_cold(file, stats)
    print("creating data set")
    training_ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "viewer": tf.cast(ratings_df["viewer"].values, tf.string),
                "viewer_gender": tf.cast(ratings_df["viewer_gender"].values, tf.string),
                "viewer_lang": tf.cast(ratings_df["viewer_lang"].values, tf.string),
                "viewer_country": tf.cast(
                    ratings_df["viewer_country"].values, tf.string
                ),
                "viewer_age": tf.cast(ratings_df["viewer_age"].values, tf.int32),
                "viewer_longitude": tf.cast(
                    ratings_df["viewer_longitude"].values, tf.float16
                ),
                "viewer_latitude": tf.cast(
                    ratings_df["viewer_latitude"].values, tf.float16
                ),
                "broadcaster": tf.cast(ratings_df["broadcaster"].values, tf.string),
                "viewer_network": tf.cast(
                    ratings_df["viewer_network"].values, tf.string
                ),
                "broadcaster_network": tf.cast(
                    ratings_df["broadcaster_network"].values, tf.string
                ),
                "viewer_lat_long": tf.cast(
                    ratings_df["viewer_lat_long"].values, tf.string
                ),
            }
        )
    )

    return training_ds


def prepare_training_data_cold(train_ds):
    print("prepare_training_data")
    training_ds = train_ds.cache().map(
        lambda x: {
            "broadcaster": x["broadcaster"],
            "viewer": x["viewer"],
            "viewer_gender": x["viewer_gender"],
            "viewer_lang": x["viewer_lang"],
            "viewer_country": x["viewer_country"],
            "viewer_age": x["viewer_age"],
            "viewer_longitude": x["viewer_longitude"],
            "viewer_latitude": x["viewer_latitude"],
            "viewer_network": x["viewer_network"],
            "broadcaster_network": x["broadcaster_network"],
            "viewer_lat_long": x["viewer_lat_long"],
        },
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    print("done prepare_training_data")
    return training_ds


# In[11]:


def get_broadcaster_data_set(train_ds):
    broadcasters = train_ds.cache().map(
        lambda x: x["broadcaster"],
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    broadcasters_ds = tf.data.Dataset.from_tensor_slices(
        np.unique(list(broadcasters.as_numpy_iterator()))
    )
    return broadcasters_ds


# In[12]:


training_dataset = load_training_data_cold(file=data_location, stats="")

# In[13]:


train = prepare_training_data_cold(training_dataset)

# In[ ]:


broadcasters_data_set = get_broadcaster_data_set(training_dataset)


# ### Prepare model conf

# In[ ]:


def get_list(training_data, key):
    return training_data.batch(1_000_000).map(
        lambda x: x[key], num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )


def get_unique_list(data):
    return np.unique(np.concatenate(list(data)))


# In[ ]:


user_genders = get_list(train, "viewer_gender")

# In[ ]:


user_langs = get_list(train, "viewer_lang")

# In[ ]:


user_countries = get_list(train, "viewer_country")

# In[ ]:


viewer_age = get_list(train, "viewer_age")

# In[ ]:


user_networks = get_list(train, "viewer_network")

# ### derive input dims

# In[ ]:


unique_user_genders = get_unique_list(user_genders)

# In[ ]:


len(unique_user_genders)

# In[ ]:


unique_user_langs = get_unique_list(user_langs)

# In[ ]:


len(unique_user_langs)

# In[ ]:


unique_user_countries = get_unique_list(user_countries)

# In[ ]:


len(unique_user_countries)

# In[ ]:


unique_user_networks = get_unique_list(user_networks)

# In[ ]:


len(unique_user_networks)

# In[ ]:


broadcaster_ids = get_list(train, "broadcaster")

# In[ ]:


unique_broadcasters = get_unique_list(broadcaster_ids)

# In[ ]:


len(unique_broadcasters)


class MultiTasks(tf.keras.Model):
    def __init__(
        self,
        candidate_model,
        query_model,
        ranking_model,
        ranking_task,
        retrieval_task,
        ranking_weight,
        retrieval_weight,
    ):
        super().__init__()

        self.query_model: tf.keras.Model = query_model
        self.candidate_model: tf.keras.Model = candidate_model
        self.ranking_model: tf.keras.Model = ranking_model
        self.ranking_task = ranking_task
        self.retrieval_task = retrieval_task

        # The loss weights.
        self.ranking_weight = ranking_weight
        self.retrieval_weight = retrieval_weight

    def train_step(self, features: Dict[Text, tf.Tensor]) -> dict:
        # Set up a gradient tape to record gradients.
        with tf.GradientTape() as tape:
            # Loss computation.

            query_embeddings = self.query_model(
                {
                    "viewer_gender": features["viewer_gender"],
                    "viewer_lang": features["viewer_lang"],
                    "viewer_country": features["viewer_country"],
                    "viewer_age": features["viewer_age"],
                    "viewer_network": features["viewer_network"],
                    "viewer_latitude": features["viewer_latitude"],
                    "viewer_longitude": features["viewer_longitude"],
                    "viewer_lat_long": features["viewer_lat_long"],
                }
            )
            positive_broadcaster_embeddings = self.candidate_model(
                features["broadcaster"]
            )
            labels = features["duration"]
            ranking_predictions = self.ranking_model(
                (query_embeddings, positive_broadcaster_embeddings)
            )
            ranking_loss = self.ranking_task(
                labels=labels,
                predictions=ranking_predictions,
            )

            retrieval_loss = self.retrieval_task(
                query_embeddings, positive_broadcaster_embeddings
            )

            regularization_loss = sum(self.losses)

            total_loss = (
                regularization_loss
                + self.ranking_weight * ranking_loss
                + self.retrieval_weight * retrieval_loss
            )

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = ranking_loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, features: Dict[Text, tf.Tensor]) -> dict:
        labels = features["duration"]

        query_embeddings = self.query_model(
            {
                "viewer_gender": features["viewer_gender"],
                "viewer_lang": features["viewer_lang"],
                "viewer_country": features["viewer_country"],
                "viewer_age": features["viewer_age"],
                "viewer_network": features["viewer_network"],
                "viewer_latitude": features["viewer_latitude"],
                "viewer_longitude": features["viewer_longitude"],
                "viewer_lat_long": features["viewer_lat_long"],
            }
        )
        positive_broadcaster_embeddings = self.candidate_model(features["broadcaster"])

        rating_predictions = self.ranking_model(
            (query_embeddings, positive_broadcaster_embeddings)
        )
        ranking_loss = self.ranking_task(labels=labels, predictions=rating_predictions)

        retrieval_loss = self.retrieval_task(
            query_embeddings, positive_broadcaster_embeddings
        )

        regularization_loss = sum(self.losses)

        total_loss = (
            regularization_loss
            + self.ranking_weight * ranking_loss
            + self.retrieval_weight * retrieval_loss
        )

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = ranking_loss + retrieval_loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss
        return metrics


# In[ ]:

cold_start_conf = CONFIG = {
    "top_k": 100,
    "broadcaster_embedding_dimension": 64,
    "batch_size": 4096,
    "learning_rate": 0.1,
    "epochs": 30,
    "patience": 3,
    "age_boundaries": [18, 25, 30, 35, 40, 45, 50, 55, 60, 65, float("inf")],
    "centroids": [
        [36.68147669256268, -82.8910274009993],
        [23.22243322909555, 78.23027450833709],
        [50.04997682638993, 0.22379313938744885],
        [37.9309447099281, -117.00741350764692],
        [-32.795864819917725, 148.7159172660312],
        [-18.570548393114084, -54.280255665692565],
        [13.921140442819565, 116.38740315555172],
        [29.78951080730802, 40.279515865947936],
    ],
}

candidate_model = CandidateModel(cold_start_conf)
query_model = QueryModel(cold_start_conf)
ranking_model = RankingModel(cold_start_conf)
ranking_task = tfrs.tasks.Ranking(
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)
retrieval_task = tfrs.tasks.Retrieval(
    metrics=tfrs.metrics.FactorizedTopK(
        candidates=broadcasters_data_set.batch(128).map(candidate_model)
    )
)

print("train model using retrieval task")
ranking_weight = 0
retrieval_weight = 1
model = MultiTasks(
    candidate_model,
    query_model,
    ranking_model,
    ranking_task,
    retrieval_task,
    ranking_weight,
    retrieval_weight,
)

tf.config.run_functions_eagerly(True)
model.compile(
    optimizer=tf.keras.optimizers.Adagrad(
        learning_rate=cold_start_conf["learning_rate"]
    ),
    run_eagerly=True,
)

shuffled = train.shuffle(100_000, seed=42, reshuffle_each_iteration=True)
train_ds = shuffled.take(80_000)
test_ds = shuffled.skip(80_000).take(20_000)
cached_train = (
    train_ds.shuffle(100_000)
    .batch(cold_start_conf["batch_size"])
    .cache()
    .prefetch(tf.data.experimental.AUTOTUNE)
)
cached_test = test_ds.batch(cold_start_conf["batch_size"]).cache()

callback = tf.keras.callbacks.EarlyStopping(
    monitor="total_loss",
    patience=cold_start_conf["patience"],
    verbose=1,
    restore_best_weights=True,
)
hist = model.fit(
    cached_train,
    epochs=cold_start_conf["epochs"],
    validation_data=cached_test,
    validation_freq=1,
    callbacks=[callback],
)

idx = np.argmin(hist.history["total_loss"])
train_accuracy = hist.history["factorized_top_k/top_100_categorical_accuracy"][idx]
print(f"Top-100 accuracy: {train_accuracy:.4f}.")

print("train model using retrieval task and ranking task")
ranking_weight = 1
retrieval_weight = 1
joint_model = MultiTasks(
    candidate_model,
    query_model,
    ranking_model,
    ranking_task,
    retrieval_task,
    ranking_weight,
    retrieval_weight,
)
joint_model.compile(
    optimizer=tf.keras.optimizers.Adagrad(
        learning_rate=cold_start_conf["learning_rate"]
    ),
    run_eagerly=True,
)
joint_model_history = joint_model.fit(
    cached_train,
    epochs=cold_start_conf["epochs"],
    validation_data=cached_test,
    validation_freq=1,
    callbacks=[callback],
)

accuracy = joint_model_history.history["factorized_top_k/top_100_categorical_accuracy"][
    -1
]
print(f"Top-100 accuracy: {accuracy:.4f}.")

idx = np.argmin(joint_model_history.history["total_loss"])
accuracy = joint_model_history.history["factorized_top_k/top_100_categorical_accuracy"][
    idx
]
print(f"Top-100 accuracy: {accuracy:.4f}.")

val_accuracy = joint_model_history.history[
    "val_factorized_top_k/top_100_categorical_accuracy"
][idx]
print(f"Top-100 accuracy: {val_accuracy:.4f}.")
