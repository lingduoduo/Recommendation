#-*-coding:utf8-*-
"""
author:linghypshen@gmail.com
produce train data for item2vec
"""

import numpy as np
from typing import Dict
from numpy import ndarray
import pandas as pd
from gensim.models import Word2Vec

def produce_train_data(input_file, out_file):
    """
    Args:
        input_file: user behavior CSV file with columns: userid, itemid, rating, timestamp
        out_file: path to write the output item vectors in Word2Vec text format
    """
    # Load data, skip first row
    df = pd.read_csv(input_file, skiprows=1, header=None, names=["userid", "itemid", "rating", "timestamp"])

    score_thr = 4.0
    # Filter: valid rows only
    df = df[df["rating"] >= score_thr]

    # Group by user and get sequences of itemids
    user_items = df.groupby("userid")["itemid"].apply(lambda x: list(map(str, x)))

    # Convert to list of lists (sentences for Word2Vec)
    sentences = user_items.tolist()

    # Train the Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=128,    # embedding size
        window=5,           # context window
        sample=1e-3,        # subsampling
        negative=5,         # negative samples
        hs=0,               # disable hierarchical softmax
        sg=1,               # skip-gram model
        epochs=50,          # number of training iterations
        workers=4           # parallel threads
    )

    # Save item vectors in text format
    model.wv.save_word2vec_format(out_file, binary=False)

def load_item_vec(input_file):
    """
    Args:
        input_file: path to item vector file
    Returns:
        dict: key=itemid, value=np.array of floats (embedding vector)
    """
    # Load the file, skipping the first line
    df = pd.read_csv(input_file, sep=" ", skiprows=1, header=None, quoting=3)

    # Expect 1 ID column + 128 dimensions = 129 columns
    expected_cols = 129
    if df.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, but got {df.shape[1]}.")

    # Remove any rows where the itemid is "</s>" (if present)
    df = df[df[0] != "</s>"]

    # Convert to dict[itemid] = np.array(vector)
    item_vec = {
        str(row[0]): np.array(row[1:], dtype=np.float32)
        for row in df.itertuples(index=False)
    }

    return item_vec

def cosine_similarity(vec_a: ndarray, vec_b: ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.
    """
    norm_product = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm_product == 0:
        return 0.0
    return np.dot(vec_a, vec_b) / norm_product

def cal_item_sim(
    item_vec: Dict[str, np.ndarray],
    itemid: str,
    output_file: str,
    topk: int = 10
) -> None:
    """
    Calculate top-k similar items for a given item and write to file.

    Args:
        item_vec (dict): {itemid: np.array of embedding vector}
        itemid (str): The fixed item ID to calculate similarity against
        output_file (str): Path to write the similar items
        topk (int): Number of top similar items to output
    """
    if itemid not in item_vec:
        print(f"[WARN] Item ID '{itemid}' not found in item_vec.")
        return

    target_vec = item_vec[itemid]

    # Create a DataFrame of other items and compute similarity
    df = pd.DataFrame([
        (other_id, cosine_similarity(target_vec, vec))
        for other_id, vec in item_vec.items() if other_id != itemid
    ], columns=["itemid", "score"])

    # Select top-k most similar items
    df_topk = df.sort_values(by="score", ascending=False).head(topk)

    # Format output string: itemid<TAB>item1_score1;item2_score2;...
    sim_str = ";".join(f"{row.itemid}_{round(row.score, 3)}" for _, row in df_topk.iterrows())
    output_line = f"{itemid}\t{sim_str}\n"

    # Write to file
    with open(output_file, "w") as f:
        f.write(output_line)

def run_main(input_file, output_file):
    item_vec = load_item_vec(input_file)
    cal_item_sim(item_vec, "27", output_file)


if __name__ == "__main__":
    # produce_train_data("../data/ratings.txt", "../data/item_vec.txt")
    # item_vec = load_item_vec("../data/item_vec.txt")
    # print(len(item_vec))
    # print(item_vec["318"])
    run_main("../data/item_vec.txt", "../data/sim_result.txt")