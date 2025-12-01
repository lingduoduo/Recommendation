#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json


import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs

local_file = "data/data_latest.csv"
if not os.path.exists(local_file) and not os.path.isfile(local_file):
    bucket = "tmg-machine-learning-models-dev"
    prefix = "for-you-payer-training-data"
    data_key = "data_latest.csv"
    data_location = "s3://{}/{}/{}".format(bucket, prefix, data_key)
else:
    data_location = local_file

# ### Model definition

# In[2]:


class UserModel(tf.keras.Model):
    def __init__(self, unique_user_ids, viewer_embedding_dimension):
        super().__init__()

        self.viewer_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=unique_user_ids, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(unique_user_ids) + 1, viewer_embedding_dimension
                ),
            ]
        )

    def call(self, inputs):
        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result.
        return tf.concat(
            [
                self.viewer_embedding(inputs["viewer"]),
            ],
            axis=1,
        )


# In[3]:


class BroadcasterModel(tf.keras.Model):
    def __init__(self, unique_broadcasters, broadcaster_embedding_dimension):
        super().__init__()

        self.broadcaster_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=unique_broadcasters, mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(unique_broadcasters) + 1, broadcaster_embedding_dimension
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


# ### Load data

# In[4]:


def load_data_file_gift(file):
    print("loading file:" + file)
    training_df = pd.read_csv(
        file,
        skiprows=[0],
        names=["broadcaster", "viewer", "count"],
        dtype={"broadcaster": np.unicode, "viewer": np.unicode, "count": np.unicode},
    )

    values = {"broadcaster": "unknown", "viewer": "unknown", "count": "unknown"}
    training_df = training_df.sample(n=10000)
    training_df.fillna(value=values, inplace=True)
    print(training_df.head(10))
    return training_df


def load_training_gift(file):
    df = load_data_file_gift(file)
    print("creating data set")
    training_ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "viewer": tf.cast(df["viewer"].values, tf.string),
                "broadcaster": tf.cast(df["broadcaster"].values, tf.string),
            }
        )
    )

    return training_ds


def prepare_training_data_gift(train_ds):
    print("prepare_training_data")
    training_ds = train_ds.map(
        lambda x: {
            "broadcaster": x["broadcaster"],
            "viewer": x["viewer"],
        },
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    return training_ds


# ```
# SELECT
#     coalesce(viewer_network||':'||viewer_member_id, 'unknown') as viewer_id,
#     coalesce(broadcaster_network||':'||broadcaster_id, 'unknown')  as broadcaster_id,
#     count(*) as count
# FROM train_data
# WHERE flag_gift = 1
# GROUP BY 1, 2
# ```

training_dataset = load_training_gift(file=data_location)

# In[6]:

train = prepare_training_data_gift(training_dataset)


# In[13]:


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


# In[14]:


broadcaster_data_set = get_broadcaster_data_set(training_dataset)


# ### Model conf

# In[15]:


def get_list(training_data, key):
    return training_data.batch(1_000_000).map(
        lambda x: x[key], num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )


def get_unique_list(data):
    return np.unique(np.concatenate(list(data)))


# In[16]:


user_ids = get_list(train, "viewer")


# In[17]:


broadcaster_ids = get_list(train, "broadcaster")


# In[18]:


data_set_size = len(broadcaster_ids)


# In[19]:


unique_broadcasters = get_unique_list(broadcaster_ids)


# In[20]:


unique_user_ids = get_unique_list(user_ids)


# In[21]:


print(len(unique_broadcasters))


# In[22]:


print(len(unique_user_ids))


# In[23]:


broadcaster_embedding_dimension = 96
viewer_embedding_dimension = 96
batch_size = 16384
learning_rate = 0.05
epochs = 3
top_k = 1000


# ### user model

# In[24]:


user_model = UserModel(unique_user_ids, viewer_embedding_dimension)


# ### broadcaster model

# In[25]:


broadcaster_model = BroadcasterModel(
    unique_broadcasters, broadcaster_embedding_dimension
)


# ### Two tower model

# In[26]:


from typing import Dict, Iterable, Tuple, Text


# In[27]:


class TwoTowers(tf.keras.Model):
    def __init__(self, broadcaster_model, user_model, task):
        super().__init__()
        self.user_model: tf.keras.Model = user_model
        self.broadcaster_model = broadcaster_model
        self.task: tf.keras.layers.Layer = task

    def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

        # Set up a gradient tape to record gradients.
        with tf.GradientTape() as tape:

            # Loss computation.

            user_embeddings = self.user_model(
                {
                    "viewer": features["viewer"],
                }
            )
            broadcaster_embeddings = self.broadcaster_model(features["broadcaster"])
            loss = self.task(user_embeddings, broadcaster_embeddings)

            # Handle regularization losses as well.
            regularization_loss = sum(self.losses)

            total_loss = loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss

        return metrics

    def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

        # Loss computation.
        user_embeddings = self.user_model(
            {
                "viewer": features["viewer"],
            }
        )
        broadcaster_embeddings = self.broadcaster_model(features["broadcaster"])
        loss = self.task(user_embeddings, broadcaster_embeddings)

        # Handle regularization losses as well.
        regularization_loss = sum(self.losses)

        total_loss = loss + regularization_loss

        metrics = {metric.name: metric.result() for metric in self.metrics}
        metrics["loss"] = loss
        metrics["regularization_loss"] = regularization_loss
        metrics["total_loss"] = total_loss
        return metrics


# In[28]:


metrics = tfrs.metrics.FactorizedTopK(
    candidates=broadcaster_data_set.batch(128).map(broadcaster_model)
)

task = tfrs.tasks.Retrieval(metrics=metrics)


# In[29]:


model = TwoTowers(broadcaster_model, user_model, task)


# In[30]:


model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=learning_rate))
cached_train = train.batch(batch_size).cache()


# In[31]:


hist = model.fit(cached_train, epochs=epochs)


# In[32]:


train_accuracy = hist.history["factorized_top_k/top_100_categorical_accuracy"][-1]

# In[33]:
print("create index")
index = tfrs.layers.factorized_top_k.BruteForce(
    query_model=user_model,
    k=top_k,
)


index.index_from_dataset(
    tf.data.Dataset.zip(
        (
            broadcaster_data_set.batch(1000),
            broadcaster_data_set.batch(1000).map(model.broadcaster_model),
        )
    )
)

# In[34]:

model_save_name = "model/train_gift_two_tower"
tf.saved_model.save(index, model_save_name)

output_conf = {
    "viewers_dim": len(unique_user_ids),
    "broadcasters_dim": len(unique_broadcasters),
    "viewer_embedding_dimension": viewer_embedding_dimension,
    "broadcaster_embedding_dimension": broadcaster_embedding_dimension,
    "train_accuracy": f"Top-100 accuracy of training: {train_accuracy:.4f}.",
    "unique_user_ids": ",".join(str(x) for x in unique_user_ids),
    "unique_broadcasters": ",".join(str(x) for x in unique_broadcasters),
}
file_path = f"{model_save_name}/model_conf.json"
print(file_path)
with open(file_path, "w", encoding="utf8") as obj_file:
    json.dump(output_conf, obj_file, separators=(",", ":"))
