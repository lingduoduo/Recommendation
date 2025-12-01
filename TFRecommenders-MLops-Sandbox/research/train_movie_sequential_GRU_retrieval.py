#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pprint
import tempfile

# In[3]:


from typing import Dict, Text

# In[4]:


import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# In[17]:


from absl import flags

# #### Preparing the dataset

# leverage the data generation utility in this TensorFlow Lite On-device Recommendation reference app.

# MovieLens 1M data contains ratings.dat (columns: UserID, MovieID, Rating, Timestamp), and movies.dat (columns: MovieID, Title, Genres). The example generation script download the 1M dataset, takes both files, only keep ratings higher than 2, form user movie interaction timelines, sample activities as labels and 10 previous user activities as the context for prediction.

# In[35]:


import collections
import json
import os
import random
import re

from absl import app
from absl import flags
from absl import logging
import pandas as pd
import tensorflow as tf

# In[6]:


# Permalinks to download movielens data.
MOVIELENS_1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
MOVIELENS_ZIP_FILENAME = "ml-1m.zip"
MOVIELENS_ZIP_HASH = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"
MOVIELENS_EXTRACTED_DIR = "ml-1m"
RATINGS_FILE_NAME = "ratings.dat"
MOVIES_FILE_NAME = "movies.dat"
RATINGS_DATA_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
MOVIES_DATA_COLUMNS = ["MovieID", "Title", "Genres"]
OUTPUT_TRAINING_DATA_FILENAME = "train_movielens_1m.tfrecord"
OUTPUT_TESTING_DATA_FILENAME = "test_movielens_1m.tfrecord"
OUTPUT_MOVIE_VOCAB_FILENAME = "movie_vocab.json"
OUTPUT_MOVIE_YEAR_VOCAB_FILENAME = "movie_year_vocab.txt"
OUTPUT_MOVIE_GENRE_VOCAB_FILENAME = "movie_genre_vocab.txt"
OUTPUT_MOVIE_TITLE_UNIGRAM_VOCAB_FILENAME = "movie_title_unigram_vocab.txt"
OUTPUT_MOVIE_TITLE_BIGRAM_VOCAB_FILENAME = "movie_title_bigram_vocab.txt"
PAD_MOVIE_ID = 0
PAD_RATING = 0.0
PAD_MOVIE_YEAR = 0
UNKNOWN_STR = "UNK"
VOCAB_MOVIE_ID_INDEX = 0
VOCAB_COUNT_INDEX = 3


# In[18]:


def define_flags():
    """Define flags."""
    flags.DEFINE_string(
        "data_dir", "/tmp", "Path to download and store movielens data."
    )
    flags.DEFINE_string("output_dir", None, "Path to the directory of output files.")
    flags.DEFINE_bool("build_vocabs", True, "If yes, generate movie feature vocabs.")
    flags.DEFINE_integer(
        "min_timeline_length", 3, "The minimum timeline length to construct examples."
    )
    flags.DEFINE_integer(
        "max_context_length", 10, "The maximum length of user context history."
    )
    flags.DEFINE_integer(
        "max_context_movie_genre_length",
        10,
        "The maximum length of user context history.",
    )
    flags.DEFINE_integer(
        "min_rating",
        None,
        "Minimum rating of movie that will be used to in " "training data",
    )
    flags.DEFINE_float("train_data_fraction", 0.9, "Fraction of training data.")


# In[19]:


define_flags()

# In[26]:


FLAGS = flags.FLAGS

# In[27]:


# !python -m example_generation_movielens
# --data_dir=data/raw
# --output_dir=data/examples
# --min_timeline_length=3
# --max_context_length=10
# --max_context_movie_genre_length=10
# --min_rating=2
# --train_data_fraction=0.9
# --build_vocabs=False
output_dir = "data/examples"
min_timeline_length = 3
max_context_length = 10
max_context_movie_genre_length = 10
min_rating = 2
build_vocabs = False
train_data_fraction = 0.9


# In[7]:


def download_and_extract_data(
    data_directory,
    url=MOVIELENS_1M_URL,
    fname=MOVIELENS_ZIP_FILENAME,
    file_hash=MOVIELENS_ZIP_HASH,
    extracted_dir_name=MOVIELENS_EXTRACTED_DIR,
):
    """Download and extract zip containing MovieLens data to a given directory.

    Args:
      data_directory: Local path to extract dataset to.
      url: Direct path to MovieLens dataset .zip file. See constants above for
        examples.
      fname: str, zip file name to download.
      file_hash: str, SHA-256 file hash.
      extracted_dir_name: str, extracted dir name under data_directory.

    Returns:
      Downloaded and extracted data file directory.
    """
    if not tf.io.gfile.exists(data_directory):
        tf.io.gfile.makedirs(data_directory)
    path_to_zip = tf.keras.utils.get_file(
        fname=fname,
        origin=url,
        file_hash=file_hash,
        hash_algorithm="sha256",
        extract=True,
        cache_dir=data_directory,
    )
    extracted_file_dir = os.path.join(os.path.dirname(path_to_zip), extracted_dir_name)
    return extracted_file_dir


# In[8]:


extracted_data_dir = download_and_extract_data(data_directory="/tmp")


# In[32]:


def read_data(data_directory, min_rating=None):
    """Read movielens ratings.dat and movies.dat file into dataframe."""
    ratings_df = pd.read_csv(
        os.path.join(data_directory, RATINGS_FILE_NAME),
        sep="::",
        names=RATINGS_DATA_COLUMNS,
        encoding="unicode_escape",
    )  # May contain unicode. Need to escape.
    ratings_df["Timestamp"] = ratings_df["Timestamp"].apply(int)
    if min_rating is not None:
        ratings_df = ratings_df[ratings_df["Rating"] >= min_rating]
    movies_df = pd.read_csv(
        os.path.join(data_directory, MOVIES_FILE_NAME),
        sep="::",
        names=MOVIES_DATA_COLUMNS,
        encoding="unicode_escape",
    )  # May contain unicode. Need to escape.
    return ratings_df, movies_df


# In[42]:


def generate_datasets(
    extracted_data_dir,
    output_dir,
    min_timeline_length,
    max_context_length,
    max_context_movie_genre_length,
    min_rating=None,
    build_vocabs=True,
    train_data_fraction=0.9,
    train_filename=OUTPUT_TRAINING_DATA_FILENAME,
    test_filename=OUTPUT_TESTING_DATA_FILENAME,
    vocab_filename=OUTPUT_MOVIE_VOCAB_FILENAME,
    vocab_year_filename=OUTPUT_MOVIE_YEAR_VOCAB_FILENAME,
    vocab_genre_filename=OUTPUT_MOVIE_GENRE_VOCAB_FILENAME,
):
    """Generates train and test datasets as TFRecord, and returns stats."""
    #     logging.info("Reading data to dataframes.")
    ratings_df, movies_df = read_data(extracted_data_dir, min_rating=min_rating)
    #     logging.info("Generating movie rating user timelines.")
    timelines, movie_counts = convert_to_timelines(ratings_df)
    #     logging.info("Generating train and test examples.")
    train_examples, test_examples = generate_examples_from_timelines(
        timelines=timelines,
        movies_df=movies_df,
        min_timeline_len=min_timeline_length,
        max_context_len=max_context_length,
        max_context_movie_genre_len=max_context_movie_genre_length,
        train_data_fraction=train_data_fraction,
    )

    if not tf.io.gfile.exists(output_dir):
        tf.io.gfile.makedirs(output_dir)
    #     logging.info("Writing generated training examples.")
    train_file = os.path.join(output_dir, train_filename)
    train_size = write_tfrecords(tf_examples=train_examples, filename=train_file)
    #     logging.info("Writing generated testing examples.")
    test_file = os.path.join(output_dir, test_filename)
    test_size = write_tfrecords(tf_examples=test_examples, filename=test_file)
    stats = {
        "train_size": train_size,
        "test_size": test_size,
        "train_file": train_file,
        "test_file": test_file,
    }

    if build_vocabs:
        (
            movie_vocab,
            movie_year_vocab,
            movie_genre_vocab,
        ) = generate_movie_feature_vocabs(
            movies_df=movies_df, movie_counts=movie_counts
        )
        vocab_file = os.path.join(output_dir, vocab_filename)
        write_vocab_json(movie_vocab, filename=vocab_file)
        stats.update(
            {
                "vocab_size": len(movie_vocab),
                "vocab_file": vocab_file,
                "vocab_max_id": max([arr[VOCAB_MOVIE_ID_INDEX] for arr in movie_vocab]),
            }
        )

        for vocab, filename, key in zip(
            [movie_year_vocab, movie_genre_vocab],
            [vocab_year_filename, vocab_genre_filename],
            ["year_vocab", "genre_vocab"],
        ):
            vocab_file = os.path.join(output_dir, filename)
            write_vocab_txt(vocab, filename=vocab_file)
            stats.update(
                {
                    key + "_size": len(vocab),
                    key + "_file": vocab_file,
                }
            )

    return stats


# In[43]:
def convert_to_timelines(ratings_df):
    """Convert ratings data to user."""
    timelines = collections.defaultdict(list)
    movie_counts = collections.Counter()
    for user_id, movie_id, rating, timestamp in ratings_df.values:
        timelines[user_id].append(
            MovieInfo(movie_id=movie_id, timestamp=int(timestamp), rating=rating)
        )
        movie_counts[movie_id] += 1
    # Sort per-user timeline by timestamp
    for (user_id, context) in timelines.items():
        context.sort(key=lambda x: x.timestamp)
        timelines[user_id] = context
    return timelines, movie_counts


# In[44]:


class MovieInfo(
    collections.namedtuple(
        "MovieInfo", ["movie_id", "timestamp", "rating", "title", "genres"]
    )
):
    """Data holder of basic information of a movie."""

    __slots__ = ()

    def __new__(
        cls, movie_id=PAD_MOVIE_ID, timestamp=0, rating=PAD_RATING, title="", genres=""
    ):
        return super(MovieInfo, cls).__new__(
            cls, movie_id, timestamp, rating, title, genres
        )


# In[48]:


def generate_examples_from_timelines(
    timelines,
    movies_df,
    min_timeline_len=3,
    max_context_len=100,
    max_context_movie_genre_len=320,
    train_data_fraction=0.9,
    random_seed=None,
    shuffle=True,
):
    """Convert user timelines to tf examples.

    Convert user timelines to tf examples by adding all possible context-label
    pairs in the examples pool.

    Args:
      timelines: The user timelines to process.
      movies_df: The dataframe of all movies.
      min_timeline_len: The minimum length of timeline. If the timeline length is
        less than min_timeline_len, empty examples list will be returned.
      max_context_len: The maximum length of the context. If the context history
        length is less than max_context_length, features will be padded with
        default values.
      max_context_movie_genre_len: The length of movie genre feature.
      train_data_fraction: Fraction of training data.
      random_seed: Seed for randomization.
      shuffle: Whether to shuffle the examples before splitting train and test
        data.

    Returns:
      train_examples: TF example list for training.
      test_examples: TF example list for testing.
    """
    examples = []
    movies_dict = generate_movies_dict(movies_df)
    progress_bar = tf.keras.utils.Progbar(len(timelines))
    for timeline in timelines.values():
        if len(timeline) < min_timeline_len:
            progress_bar.add(1)
            continue
        single_timeline_examples = generate_examples_from_single_timeline(
            timeline=timeline,
            movies_dict=movies_dict,
            max_context_len=max_context_len,
            max_context_movie_genre_len=max_context_movie_genre_len,
        )
        examples.extend(single_timeline_examples)
        progress_bar.add(1)
    # Split the examples into train, test sets.
    if shuffle:
        random.seed(random_seed)
        random.shuffle(examples)
    last_train_index = round(len(examples) * train_data_fraction)

    train_examples = examples[:last_train_index]
    test_examples = examples[last_train_index:]
    return train_examples, test_examples


# In[50]:
def generate_movies_dict(movies_df):
    """Generates movies dictionary from movies dataframe."""
    movies_dict = {
        movie_id: MovieInfo(movie_id=movie_id, title=title, genres=genres)
        for movie_id, title, genres in movies_df.values
    }
    movies_dict[0] = MovieInfo()
    return movies_dict


# In[52]:


def generate_examples_from_single_timeline(
    timeline, movies_dict, max_context_len=100, max_context_movie_genre_len=320
):
    """Generate TF examples from a single user timeline.

    Generate TF examples from a single user timeline. Timeline with length less
    than minimum timeline length will be skipped. And if context user history
    length is shorter than max_context_len, features will be padded with default
    values.

    Args:
      timeline: The timeline to generate TF examples from.
      movies_dict: Dictionary of all MovieInfos.
      max_context_len: The maximum length of the context. If the context history
        length is less than max_context_length, features will be padded with
        default values.
      max_context_movie_genre_len: The length of movie genre feature.

    Returns:
      examples: Generated examples from this single timeline.
    """
    examples = []
    for label_idx in range(1, len(timeline)):
        start_idx = max(0, label_idx - max_context_len)
        context = timeline[start_idx:label_idx]
        # Pad context with out-of-vocab movie id 0.
        while len(context) < max_context_len:
            context.append(MovieInfo())
        label_movie_id = int(timeline[label_idx].movie_id)
        context_movie_id = [int(movie.movie_id) for movie in context]
        context_movie_rating = [movie.rating for movie in context]
        context_movie_year = generate_feature_of_movie_years(movies_dict, context)
        context_movie_genres = generate_movie_genres(movies_dict, context)
        context_movie_genres = _pad_or_truncate_movie_feature(
            context_movie_genres,
            max_context_movie_genre_len,
            tf.compat.as_bytes(UNKNOWN_STR),
        )
        feature = {
            "context_movie_id": tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_movie_id)
            ),
            "context_movie_rating": tf.train.Feature(
                float_list=tf.train.FloatList(value=context_movie_rating)
            ),
            "context_movie_genre": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=context_movie_genres)
            ),
            "context_movie_year": tf.train.Feature(
                int64_list=tf.train.Int64List(value=context_movie_year)
            ),
            "label_movie_id": tf.train.Feature(
                int64_list=tf.train.Int64List(value=[label_movie_id])
            ),
        }
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        examples.append(tf_example)

    return examples


# In[54]:


def generate_feature_of_movie_years(movies_dict, movies):
    """Extracts year feature for movies from movie title."""
    return [
        extract_year_from_title(movies_dict[movie.movie_id].title) for movie in movies
    ]


# In[56]:


def extract_year_from_title(title):
    year = re.search(r"\((\d{4})\)", title)
    if year:
        return int(year.group(1))
    return 0


# In[58]:


def generate_movie_genres(movies_dict, movies):
    """Create a feature of the genre of each movie.

    Save genre as a feature for the movies.

    Args:
      movies_dict: Dict of movies, keyed by movie_id with value of (title, genre)
      movies: list of movies to extract genres.

    Returns:
      movie_genres: list of genres of all input movies.
    """
    movie_genres = []
    for movie in movies:
        if not movies_dict[movie.movie_id].genres:
            continue
        genres = [
            tf.compat.as_bytes(genre)
            for genre in movies_dict[movie.movie_id].genres.split("|")
        ]
        movie_genres.extend(genres)

    return movie_genres


# In[60]:


def _pad_or_truncate_movie_feature(feature, max_len, pad_value):
    feature.extend([pad_value for _ in range(max_len - len(feature))])
    return feature[:max_len]


# In[62]:


def write_tfrecords(tf_examples, filename):
    """Writes tf examples to tfrecord file, and returns the count."""
    with tf.io.TFRecordWriter(filename) as file_writer:
        length = len(tf_examples)
        progress_bar = tf.keras.utils.Progbar(length)
        for example in tf_examples:
            file_writer.write(example.SerializeToString())
            progress_bar.add(1)
        return length


# In[63]:


# stats = generate_datasets(
# 	extracted_data_dir = extracted_data_dir,
# 	output_dir = output_dir,
# 	min_timeline_length = min_timeline_length,
# 	max_context_length = max_context_length,
# 	max_context_movie_genre_length = max_context_movie_genre_length,
# 	min_rating = min_rating,
# 	build_vocabs = build_vocabs,
# 	train_data_fraction = train_data_fraction
# )

# In[64]:


# stats


# Here is a sample of the generated dataset.
#
# ```0 : {
#   features: {
#     feature: {
#       key  : "context_movie_id"
#       value: { int64_list: { value: [ 1124, 2240, 3251, ..., 1268 ] } }
#     }
#     feature: {
#       key  : "context_movie_rating"
#       value: { float_list: {value: [ 3.0, 3.0, 4.0, ..., 3.0 ] } }
#     }
#     feature: {
#       key  : "context_movie_year"
#       value: { int64_list: { value: [ 1981, 1980, 1985, ..., 1990 ] } }
#     }
#     feature: {
#       key  : "context_movie_genre"
#       value: { bytes_list: { value: [ "Drama", "Drama", "Mystery", ..., "UNK" ] } }
#     }
#     feature: {
#       key  : "label_movie_id"
#       value: { int64_list: { value: [ 3252 ] }  }
#     }
#   }
# }
# ```

# In[65]:


def _parse_function(example_proto):
    return tf.io.parse_single_example(example_proto, feature_description)


# In[66]:


train_filename = "./data/examples/train_movielens_1m.tfrecord"
train = tf.data.TFRecordDataset(train_filename)

test_filename = "./data/examples/test_movielens_1m.tfrecord"
test = tf.data.TFRecordDataset(test_filename)

# In[67]:


feature_description = {
    "context_movie_id": tf.io.FixedLenFeature(
        [10], tf.int64, default_value=np.repeat(0, 10)
    ),
    "context_movie_rating": tf.io.FixedLenFeature(
        [10], tf.float32, default_value=np.repeat(0, 10)
    ),
    "context_movie_year": tf.io.FixedLenFeature(
        [10], tf.int64, default_value=np.repeat(1980, 10)
    ),
    "context_movie_genre": tf.io.FixedLenFeature(
        [10], tf.string, default_value=np.repeat("Drama", 10)
    ),
    "label_movie_id": tf.io.FixedLenFeature([1], tf.int64, default_value=0),
}

train_ds = (
    train.map(_parse_function)
    .map(
        lambda x: {
            "context_movie_id": tf.strings.as_string(x["context_movie_id"]),
            "label_movie_id": tf.strings.as_string(x["label_movie_id"]),
        }
    )
    .take(1000)
)

test_ds = (
    test.map(_parse_function)
    .map(
        lambda x: {
            "context_movie_id": tf.strings.as_string(x["context_movie_id"]),
            "label_movie_id": tf.strings.as_string(x["label_movie_id"]),
        }
    )
    .take(1000)
)

# In[69]:


for x in train_ds.take(2).as_numpy_iterator():
    pprint.pprint(x)

# In[70]:


movies = tfds.load("movielens/1m-movies", split="train")
movies = movies.map(lambda x: x["movie_id"])
movie_ids = movies.batch(1_000)
unique_movie_ids = np.unique(np.concatenate(list(movie_ids)))

# ### Implementing a sequential model

# In our basic retrieval tutorial, we use one query tower for the user, and the candidate tow for the candidate movie. However, the two-tower architecture is generalizble and not limited to <user,item> pair. You can also use it to do item-to-item recommendation as we note in the basic retrieval tutorial.
#
# Here we are still going to use the two-tower architecture. Specificially, we use the query tower with a Gated Recurrent Unit (GRU) layer to encode the sequence of historical movies, and keep the same candidate tower for the candidate movie.

# In[71]:


embedding_dimension = 32

query_model = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_movie_ids, mask_token=None
        ),
        tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension),
        tf.keras.layers.GRU(embedding_dimension),
    ]
)

candidate_model = tf.keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=unique_movie_ids, mask_token=None
        ),
        tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension),
    ]
)

# In[72]:


metrics = tfrs.metrics.FactorizedTopK(candidates=movies.batch(128).map(candidate_model))

task = tfrs.tasks.Retrieval(metrics=metrics)


class Model(tfrs.Model):
    def __init__(self, query_model, candidate_model):
        super().__init__()
        self._query_model = query_model
        self._candidate_model = candidate_model

        self._task = task

    def compute_loss(self, features, training=False):
        watch_history = features["context_movie_id"]
        watch_next_label = features["label_movie_id"]

        query_embedding = self._query_model(watch_history)
        candidate_embedding = self._candidate_model(watch_next_label)

        return self._task(
            query_embedding, candidate_embedding, compute_metrics=not training
        )


# ### Fitting and evaluating

# In[73]:


model = Model(query_model, candidate_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# In[74]:


cached_train = train_ds.shuffle(10_00).batch(1280).cache()
cached_test = test_ds.batch(256).cache()

# In[75]:


hist = model.fit(cached_train, epochs=1)

# In[76]:

# model.evaluate(cached_test, return_dict = True)

train_accuracy = hist.history["factorized_top_k/top_100_categorical_accuracy"][-1]
print(train_accuracy)

# In[34]:

# Use brute-force search to set up retrieval using the trained representations.
# index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# index.index_from_dataset(
#     movies.batch(100).map(lambda title: (title, model.movie_model(title))))

# Get some recommendations.
# _, titles = index(np.array(["42"]))
# print(f"Top 3 recommendations for user 42: {titles[0, :3]}")

# model_save_name = "model/train_movie_GRU"
# tf.saved_model.save(model, model_save_name)
