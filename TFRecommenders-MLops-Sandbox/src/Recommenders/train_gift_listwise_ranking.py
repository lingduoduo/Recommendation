#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import re
from typing import Dict, Text
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_ranking as tfr
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
    def __init__(self, user_conf):
        super().__init__()

        self.viewer_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=user_conf["unique_user_ids"], mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(user_conf["unique_user_ids"]) + 1,
                    user_conf["viewer_embedding_dimension"],
                ),
            ]
        )

    def call(self, inputs):
        return tf.concat(
            [
                self.viewer_embedding(inputs["viewer"]),
            ],
            1,
        )


# In[3]:


class BroadcasterModel(tf.keras.Model):
    def __init__(self, broadcaster_conf):
        super().__init__()

        self.broadcaster_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=broadcaster_conf["unique_broadcasters"], mask_token=None
                ),
                tf.keras.layers.Embedding(
                    len(broadcaster_conf["unique_broadcasters"]) + 1,
                    broadcaster_conf["broadcaster_embedding_dimension"],
                ),
            ]
        )

    def call(self, broadcaster):
        return tf.concat(
            [
                self.broadcaster_embedding(broadcaster),
            ],
            1,
        )


### Load data


def load_data_file_gift(file):
    print("loading file:" + file)
    training_df = pd.read_csv(
        file,
        skiprows=[0],
        names=["broadcaster", "viewer", "count"],
        dtype={"broadcaster": np.unicode, "viewer": np.unicode, "count": np.float32},
    )

    values = {"broadcaster": "unknown", "viewer": "unknown", "count": "unknown"}
    training_df = training_df.sample(n=10000)
    training_df.fillna(value=values, inplace=True)
    print(training_df.head(10))
    return training_df


def load_training_gift(file):
    load_df = load_data_file_gift(file)
    print("creating data set")
    training_ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "viewer": tf.cast(load_df["viewer"].values, tf.string),
                "broadcaster": tf.cast(load_df["broadcaster"].values, tf.string),
                "count": tf.cast(load_df["count"].values, tf.float32),
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
            "count": x["count"],
        },
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )

    return training_ds


training_dataset = load_training_gift(file=data_location)

# In[6]:

train = prepare_training_data_gift(training_dataset)


# In[13]:


def get_broadcaster_data_set(train_ds):
    get_broadcasters = train_ds.cache().map(
        lambda x: x["broadcaster"],
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    broadcasters_ds = tf.data.Dataset.from_tensor_slices(
        np.unique(list(get_broadcasters.as_numpy_iterator()))
    )
    return broadcasters_ds


def get_list(training_data, key):
    return training_data.batch(1_000_000).map(
        lambda x: x[key], num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
    )


def get_unique_list(data):
    return np.unique(np.concatenate(list(data)))


# ### Model conf

# In[14]:


broadcaster_data_set = get_broadcaster_data_set(training_dataset)

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

conf = dict()
conf["unique_user_ids"] = unique_user_ids
conf["unique_broadcasters"] = unique_broadcasters
conf["broadcaster_embedding_dimension"] = 96
conf["viewer_embedding_dimension"] = 96
batch_size = 16384
learning_rate = 0.05
epochs = 3
top_k = 1000

# ### user model

# In[24]:


user_model = UserModel(conf)

# ### broadcaster model

# In[25]:


broadcaster_model = BroadcasterModel(conf)


# ### Two tower model


class TwoTowers(tf.keras.Model):
    def __init__(
        self, two_tower_broadcaster_model, two_tower_user_model, two_tower_task
    ):
        super().__init__()
        self.user_model: tf.keras.Model = two_tower_user_model
        self.broadcaster_model = two_tower_broadcaster_model
        self.task: tf.keras.layers.Layer = two_tower_task

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

        two_tower_metrics = {metric.name: metric.result() for metric in self.metrics}
        two_tower_metrics["loss"] = loss
        two_tower_metrics["regularization_loss"] = regularization_loss
        two_tower_metrics["total_loss"] = total_loss

        return two_tower_metrics

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

        two_tower_metrics = {metric.name: metric.result() for metric in self.metrics}
        two_tower_metrics["loss"] = loss
        two_tower_metrics["regularization_loss"] = regularization_loss
        two_tower_metrics["total_loss"] = total_loss
        return two_tower_metrics


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

_, broadcasters = index(
    {
        "viewer": tf.constant(["kik:user:iamtesla215_ju3"]),
    }
)

print(f"Recommendations for user kik:user:iamtesla215_ju3: {broadcasters}")

# In[34]:

model_save_name = "model/train_gift_two_tower"
tf.saved_model.save(index, model_save_name)

output_conf = {
    "viewers_dim": len(unique_user_ids),
    "broadcasters_dim": len(unique_broadcasters),
    "viewer_embedding_dimension": conf["viewer_embedding_dimension"],
    "broadcaster_embedding_dimension": conf["broadcaster_embedding_dimension"],
    "train_accuracy": f"Top-100 accuracy of training: {train_accuracy:.4f}.",
    "unique_user_ids": ",".join(str(x) for x in unique_user_ids),
    "unique_broadcasters": ",".join(str(x) for x in unique_broadcasters),
}
file_path = f"{model_save_name}/model_conf.json"
print(file_path)
with open(file_path, "w", encoding="utf8") as obj_file:
    json.dump(output_conf, obj_file, separators=(",", ":"))

### Ranking Model


# In[5]:

# candidate-model-lambda-normal-v2.tar.gz
retrieval_model = tf.saved_model.load("model/train_gift_two_tower")

# In[9]:


# candidate-model-lambda-gift-v2.tar.gz
json_file = open("model/train_gift_two_tower/model_conf.json")
model_config = json.load(json_file)
regex = re.compile("'(.*?)'")
unique_user_ids = regex.findall(model_config["unique_user_ids"])
unique_broadcasters = regex.findall(model_config["unique_broadcasters"])

# In[10]:


print(f"len(unique_user_ids) = {len(unique_user_ids)}")

# In[11]:


print(f"model_config['viewers_dim'] = {model_config['viewers_dim']}")

# In[12]:


print(f"len(unique_broadcasters) = {len(unique_broadcasters)}")

# In[13]:


print(f"model_config['broadcasters_dim'] = {model_config['broadcasters_dim']}")

df = load_data_file_gift(local_file)
retrieval_results = {
    "viewer": [],
    "broadcaster": [],
    "retrieval_score": [],
    "count": [],
}
for user_id in unique_user_ids:
    # for user_id in ['zoosk:de347500a97c284a84c1b14071f4c0cd', 'agged:5404088037']:
    scores, topk_broadcasters = retrieval_model(
        {
            "viewer": tf.constant([user_id]),
        },
    )
    topk_broadcasters = topk_broadcasters.numpy()[0][:100]
    scores = scores.numpy()[0][:100]
    retrieval_results["viewer"].append(user_id)
    d = dict(
        zip(
            df.loc[df["viewer"] == user_id, "broadcaster"],
            df.loc[df["viewer"] == user_id, "count"],
        )
    )
    user_score = []
    user_broadcaster = []
    user_count = []
    for v, b in zip(scores, topk_broadcasters):
        user_score.append(v)
        user_broadcaster.append(b.decode("utf-8"))
        if b.decode("utf-8") in d:
            user_count.append(d[b.decode("utf-8")])
        else:
            user_count.append(0.0)
    retrieval_results["retrieval_score"].append(user_score)
    retrieval_results["broadcaster"].append(user_broadcaster)
    retrieval_results["count"].append(user_count)

retrieval_results_by_user_ds = tf.data.Dataset.from_tensor_slices(retrieval_results)


class RankingModel(tfrs.Model):
    def __init__(self, loss):
        super().__init__()
        embedding_dimension = 32

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=unique_user_ids
                ),
                tf.keras.layers.Embedding(
                    len(unique_user_ids) + 1, embedding_dimension
                ),
            ]
        )

        # Compute embeddings for broadcasters.
        self.broadcaster_embeddings = tf.keras.Sequential(
            [
                tf.keras.layers.experimental.preprocessing.StringLookup(
                    vocabulary=unique_broadcasters
                ),
                tf.keras.layers.Embedding(
                    len(unique_broadcasters) + 1, embedding_dimension
                ),
            ]
        )

        # Compute predictions.
        self.score_model = tf.keras.Sequential(
            [
                # Learn multiple dense layers.
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                # Make rating predictions in the final layer.
                tf.keras.layers.Dense(1),
            ]
        )

        self.task = tfrs.tasks.Ranking(
            loss=loss,
            metrics=[
                tfr.keras.metrics.NDCGMetric(name="ndcg_metric"),
                tf.keras.metrics.RootMeanSquaredError(),
            ],
        )

    def call(self, features):
        user_embeddings = self.user_embeddings(features["viewer"])
        broadcaster_embeddings = self.broadcaster_embeddings(features["broadcaster"])

        list_length = features["broadcaster"].shape[1]
        user_embedding_repeated = tf.repeat(
            tf.expand_dims(user_embeddings, 1), [list_length], axis=1
        )
        concatenated_embeddings = tf.concat(
            [user_embedding_repeated, broadcaster_embeddings], 2
        )

        return self.score_model(concatenated_embeddings)

    def compute_loss(self, inputs, training=False):
        labels = inputs.pop("count")

        ranking_scores = self(inputs)

        return self.task(
            labels=labels,
            predictions=tf.squeeze(ranking_scores, axis=-1),
        )


# # In[18]:
#
#
cached_train = retrieval_results_by_user_ds.shuffle(100_000).batch(8192).cache()

# # In[19]:
#
#
mse_model = RankingModel(tf.keras.losses.MeanSquaredError())
mse_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.01))

# # In[20]:
#
#
epochs = 5

# # In[21]:
#
#
hist = mse_model.fit(cached_train, epochs=epochs, verbose=True)
pointwise_ndcg_metric = hist.history["ndcg_metric"][-1]
print(f"pointwise_ndcg_metric = {pointwise_ndcg_metric}")

# # In[22]:
#
#
hinge_model = RankingModel(tfr.keras.losses.PairwiseHingeLoss())
hinge_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

# # In[23]:
#
#
hist = hinge_model.fit(cached_train, epochs=epochs, verbose=True)
pairwise_ndcg_metric = hist.history["ndcg_metric"][-1]
print(f"pairwise_ndcg_metric = {pairwise_ndcg_metric}")

# # In[24]:
#
#
listwise_model = RankingModel(tfr.keras.losses.ListMLELoss())
listwise_model.compile(optimizer=tf.keras.optimizers.Adagrad(0.01))

# # In[25]:
#
#
hist = listwise_model.fit(cached_train, epochs=epochs, verbose=True)
listwise_ndcg_metric = hist.history["ndcg_metric"][-1]
print(f"listwise_ndcg_metric = {listwise_ndcg_metric}")

# save listwise
tf.saved_model.save(listwise_model, "model/train_listwise_ranking")

# load listwise model
loaded_model = tf.saved_model.load("model/train_listwise_ranking")

# score new dataset
test_data = {"viewer": [], "broadcaster": [], "retrieval_score": []}
for user_id in ["zoosk:de347500a97c284a84c1b14071f4c0cd", "agged:5404088037"]:
    scores, topk_broadcasters = retrieval_model(
        {
            "viewer": tf.constant([user_id]),
        }
    )
    topk_broadcasters = topk_broadcasters.numpy()[0][:100]
    scores = scores.numpy()[0][:100]
    test_data["viewer"].append(user_id)
    test_data["broadcaster"].append(topk_broadcasters)
    test_data["retrieval_score"].append(scores)
test_data_ds = tf.data.Dataset.from_tensor_slices(test_data)
test_data_ds_cached = test_data_ds.batch(8192).cache()
test_score = []
for cached_test_batch in test_data_ds_cached:
    test_score.append(loaded_model(cached_test_batch))
print(test_score)
