#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : ling.huang@adp.com
@File    : Item2Vec.py
"""
import os
import numpy as np
from typing import Dict
from numpy import ndarray
import pandas as pd
from gensim.models import Word2Vec
from pathlib import Path

# Get the project root directory using pathlib
ROOT_DIR = Path.cwd().parent.parent
local_path = ROOT_DIR / "src" / "data" / "input"

local_click_file = "search_click.csv"
local_click_path_file = local_path / local_click_file

local_item_file = "item_desc.csv"
local_item_path_file = local_path / local_item_file

local_path = ROOT_DIR / "src" / "data" / "output"
local_item_vector_file = "item_vector.csv"
local_item_vector_path_file = local_path / local_item_vector_file

local_sim_result_file = "item2vec_sim_result.csv"
local_sim_result_path_file = local_path / local_sim_result_file

score_thr = 1


def produce_train_data(input_path_file, output_path_file):
    """
    Args:
        input_path_file: user behavior CSV file with columns: userid, item_id, rating, timestamp
        out_path_file: path to write the output item vectors in Word2Vec text format
    """
    # Load data, skip first row
    df = pd.read_csv(input_path_file,
                     skiprows=1,
                     header=None,
                     names=["user_id", "item_id", "rating"],
                     dtype={"user_id": str, "item_id": str, "rating": float},
                     )

    # Filter: valid rows only
    # df = df[df["rating"] >= score_thr]

    # Group by user and get sequences of item_ids
    user_items = df.groupby("user_id")["item_id"].apply(lambda x: list(map(str, x)))

    # Convert to list of lists (sentences for Word2Vec)
    sentences = user_items.tolist()
    print("sentences: ", len(sentences))

    # Train the Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=128,  # embedding size
        window=5,  # context window
        sample=1e-3,  # subsampling
        negative=5,  # negative samples
        hs=0,  # disable hierarchical softmax
        sg=1,  # skip-gram model
        epochs=50,  # number of training iterations
        workers=4  # parallel threads
    )

    # Save item vectors in text format
    output_dir = os.path.dirname(output_path_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.wv.save_word2vec_format(output_path_file, binary=False)


def load_item_vec(input_file):
    """
    Args:
        input_file: path and file to item vector file
    Returns:
        dict: key=item_id, value=np.array of floats (embedding vector)
    """
    # Load the file, skipping the first line
    df = pd.read_csv(input_file, sep=" ", skiprows=1, header=None, quoting=3)

    # Expect 1 ID column + 128 dimensions = 129 columns
    expected_cols = 129
    if df.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, but got {df.shape[1]}.")

    # Remove any rows where the item_id is "</s>" (if present)
    df = df[df[0] != "</s>"]

    # Convert to dict[item_id] = np.array(vector)
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
        item_id: str,
        output_path_file: str,
        topk: int = 10
) -> None:
    """
    Calculate top-k similar items for a given item and write to file.

    Args:
        item_vec (dict): {item_id: np.array of embedding vector}
        item_id (str): The fixed item ID to calculate similarity against
        topk (int): Number of top similar items to output
    """
    if item_id not in item_vec:
        print(f"[WARN] Item ID '{item_id}' not found in item_vec.")
        return

    target_vec = item_vec[item_id]

    # Create a DataFrame of other items and compute similarity
    df = pd.DataFrame([
        (other_id, cosine_similarity(target_vec, vec))
        for other_id, vec in item_vec.items() if other_id != item_id
    ], columns=["item_id", "score"])

    # Select top-k most similar items
    df_topk = df.sort_values(by="score", ascending=False).head(topk)

    # Format output string: item_id<TAB>item1_score1;item2_score2;...
    sim_str = ";".join(f"{row.item_id}_{round(row.score, 3)}" for _, row in df_topk.iterrows())
    output_line = f"{item_id}\t{sim_str}\n"

    # Write to file
    output_dir = os.path.dirname(output_path_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path_file, "w") as f:
        f.write(output_line)


def run_main(input_path_file, output_path_file):
    item_vec = load_item_vec(input_path_file)
    cal_item_sim(item_vec, "b3cc3ceac4d24c2e843aa13078bd2f8e", output_path_file)


if __name__ == "__main__":
    produce_train_data(local_click_path_file, local_item_vector_path_file)

    # item_vec = load_item_vec( local_item_vector_path_file)
    run_main(local_item_vector_path_file, local_sim_result_path_file)
