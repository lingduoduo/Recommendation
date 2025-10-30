# in case this is run outside of conda environment with python2
import itertools
import os
import pprint

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_recommenders as tfrs


def printds(ds, n=5):
    for example in ds.take(n):
        pprint.pprint(example)


class DCN(tfrs.Model):
    def __init__(
        self, dcn_conf, use_cross_layer, deep_layer_sizes, projection_dim=None
    ):
        super().__init__()

        self.embedding_dimension = dcn_conf["embedding_dimension"]
        str_features = dcn_conf["str_features"]
        int_features = dcn_conf["int_features"]
        self._all_features = str_features + int_features
        self._embeddings = {}
        self.label_name = dcn_conf["label_name"]

        # Compute embeddings for string features.
        input_vocabularies = dcn_conf["vocabularies"]
        for str_feature_name in str_features:
            vocabulary = input_vocabularies[str_feature_name]
            self._embeddings[str_feature_name] = tf.keras.Sequential(
                [
                    tf.keras.layers.experimental.preprocessing.StringLookup(
                        vocabulary=vocabulary, mask_token=None
                    ),
                    tf.keras.layers.Embedding(
                        len(vocabulary) + 1,
                        # self.embedding_dimension
                        6 * int(pow(len(vocabulary), 0.25)),
                    ),
                ]
            )

        # Compute embeddings for int features.
        for int_feature_name in int_features:
            vocabulary = vocabularies[int_feature_name]
            self._embeddings[int_feature_name] = tf.keras.Sequential(
                [
                    tf.keras.layers.IntegerLookup(
                        vocabulary=vocabulary, mask_value=None
                    ),
                    tf.keras.layers.Embedding(
                        len(vocabulary) + 1,
                        # self.embedding_dimension
                        6 * int(pow(len(vocabulary), 0.25)),
                    ),
                ]
            )

        if use_cross_layer:
            self._cross_layer = tfrs.layers.dcn.Cross(
                projection_dim=projection_dim, kernel_initializer="glorot_uniform"
            )
        else:
            self._cross_layer = None

        self._deep_layers = [
            tf.keras.layers.Dense(layer_size, activation="relu")
            for layer_size in deep_layer_sizes
        ]

        self._logit_layer = tf.keras.layers.Dense(1)

        self.task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")],
        )

    def call(self, inputs):
        # Concatenate embeddings
        embeddings = []
        for all_feature_name in self._all_features:
            embedding_fn = self._embeddings[all_feature_name]
            embeddings.append(embedding_fn(inputs[all_feature_name]))

        x = tf.concat(embeddings, 1)

        # Build Cross Network
        if self._cross_layer is not None:
            x = self._cross_layer(x)

        # Build Deep Network
        for deep_layer in self._deep_layers:
            x = deep_layer(x)

        return self._logit_layer(x)

    def compute_loss(self, inputs, training=False):
        labels = inputs.pop(self.label_name)
        scores = self(inputs)
        return self.task(
            labels=labels,
            predictions=scores,
        )


def load_data_file_gift(file):
    print("loading file:" + file)
    training_df = pd.read_csv(
        file,
        skiprows=[0],
        names=["broadcaster", "viewer", "product_name", "order_time", "count"],
        dtype={
            "broadcaster": np.unicode,
            "viewer": np.unicode,
            "product_name": np.unicode,
            "order_time": np.unicode,
            "count": np.int,
        },
    )

    values = {
        "broadcaster": "unknown",
        "viewer": "unknown",
        "product_name": "unknown",
        "order_time": "0",
        "count": 0,
    }

    training_df = training_df.sample(n=10000)
    training_df.fillna(value=values, inplace=True)
    return training_df


def load_training_gift(file):
    df = load_data_file_gift(file)
    print("creating data set")
    training_ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "viewer": tf.cast(df["viewer"].values, tf.string),
                "broadcaster": tf.cast(df["broadcaster"].values, tf.string),
                "product_name": tf.cast(df["product_name"].values, tf.string),
                "order_time": tf.cast(df["order_time"].values, tf.string),
                "count": tf.cast(df["count"].values, tf.int64),
            }
        )
    )
    return training_ds, len(df)


def prepare_training_data_gift(train_ds):
    print("prepare_training_data")
    training_ds = train_ds.map(
        lambda x: {
            "broadcaster": x["broadcaster"],
            "viewer": x["viewer"],
            "product_name": x["product_name"],
            "order_time": x["order_time"],
            "count": x["count"],
        },
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return training_ds


def feature_mapping(train_ds, feature_name):
    vocab = train_ds.batch(1_000_000).map(
        lambda x: x[feature_name],
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    return np.unique(np.concatenate(list(vocab)))


# Fetch the data
# local_file = "data/data_latest.csv"
local_file = "data/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
if not os.path.exists(local_file) and not os.path.isfile(local_file):
    bucket = "tmg-machine-learning-models-dev"
    prefix = "for-you-payer-training-data"
    data_key = "data/65cb05a3-e45a-4a15-915b-90cf082dc203.csv"
    data_location = "s3://{}/{}/{}".format(bucket, prefix, data_key)
else:
    data_location = local_file

dataset, nrow = load_training_gift(local_file)
print(f"nrow:{nrow}")
gift = prepare_training_data_gift(dataset)
shuffled = gift.shuffle(nrow, seed=42, reshuffle_each_iteration=False)

conf = {
    "batch_size": 16384,
    "learning_rate": 0.05,
    "epochs": 3,
    "top_k": 100,
    "embedding_dimension": 96,
    "str_features": ["broadcaster", "viewer", "product_name", "order_time"],
    "int_features": [],
    "label_name": "count",
}
ds_train = shuffled.take(int(nrow * 0.8))
ds_train = ds_train.cache()
ds_train = ds_train.batch(conf["batch_size"])
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = shuffled.skip(int(nrow * 0.8)).take(int(nrow * 0.2))
ds_test = ds_test.batch(conf["batch_size"]).cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

# Fetch feature and vocabularies
features = ["viewer", "broadcaster", "product_name", "order_time"]
vocabularies = {}
for idx, feature in enumerate(features):
    print(f"{idx}: {feature}")
    vocabularies[feature] = feature_mapping(gift, feature)
conf["vocabularies"] = vocabularies

# Train the Model.
model = DCN(
    dcn_conf=conf,
    use_cross_layer=True,
    deep_layer_sizes=[192, 192],
    projection_dim=None,
)
model.compile(optimizer=tf.keras.optimizers.Adam(conf["learning_rate"]))
model.fit(ds_train, epochs=conf["epochs"], verbose=False)
metrics = model.evaluate(ds_test, return_dict=True)
print(f"metrics: {metrics}")

# saved dcm model
tf.saved_model.save(model, "model/train_dcn_ranking")

# load model
loaded_model = tf.saved_model.load("model/train_dcn_ranking")
score = loaded_model(
    {
        "viewer": np.array(["unknown"]),
        "broadcaster": np.array(["unknown"]),
        "product_name": np.array(["unknown"]),
        "order_time": np.array(["1"]),
    }
).numpy()
print(score)

scoring_df = pd.read_csv("data/65cb05a3-e45a-4a15-915b-90cf082dc203.csv")
pred_viewer_id = np.unique(scoring_df["viewer_id"].values)[:1]
pred_broadcaster_id = np.unique(scoring_df["broadcaster_id"].values)[:1]
product_name = np.unique(scoring_df["product_name"].values)[:1]
ordered_time = np.unique(scoring_df["ordered_time"].values)[:1]
prediction_df = pd.DataFrame(
    list(
        itertools.product(
            pred_viewer_id, pred_broadcaster_id, product_name, ordered_time
        )
    )
)
prediction_df.columns = ["viewer", "broadcaster", "product_name", "order_time"]
prediction_df["scores"] = prediction_df.apply(
    lambda x: loaded_model(
        {
            "viewer": np.array([x["viewer"]]),
            "broadcaster": [x["broadcaster"]],
            "product_name": [x["product_name"]],
            "order_time": [str(x["order_time"])],
        }
    ).numpy()[0],
    axis=1,
)
print(prediction_df)
