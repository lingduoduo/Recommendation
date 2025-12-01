#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : linghypshen@gmail.com
@File    : LFM.py
"""
import numpy as np
import operator
import os
import pandas as pd
import random
from pathlib import Path

# Get the project root directory using pathlib
ROOT_DIR = Path.cwd().parent.parent
local_path = ROOT_DIR / "src" / "data" / "input"

local_click_file = "search_click.csv"
local_click_path_file = local_path / local_click_file

local_item_file = "item_desc.csv"
local_item_path_file = local_path / local_item_file
score_thr = 1


def get_item_info(input_path_file):
    """
    Load item info using pandas.
    Args:
        input_path_file (str): Path and file name of the input file
    Returns:
         dictionary: key = itemid, value = [title]
    """
    df = pd.read_csv(input_path_file,
                     skiprows=1,
                     header=None,
                     names=["item_id", "title"],
                     dtype={"item_id": str, "title": str},
                     )
    item_info = dict(zip(df['item_id'], zip(df['title'])))
    return item_info


def get_ave_score(input_path_file):
    """
    Compute average rating score per item using pandas.
    Args:
        input_path_file (str): Path and file name of the input file
    Returns:
        dictionary: key = item_id, value = average score (rounded to 3 decimals)
    """

    # Load the file with pandas, skip bad lines if any
    df = pd.read_csv(input_path_file,
                     skiprows=1,
                     header=None,
                     names=["user_id", "item_id", "rating"],
                     dtype={"user_id": str, "item_id": str, "rating": float},
                     )
    # print(f"avg rating: {df['rating'].mean()}")
    score_series = df.groupby('item_id')['rating'].mean().round(3)
    return score_series.to_dict()


def get_train_data(input_path_file):
    """
    Generate training data for LFM model using pandas.
    Args:
        input_path_file (str): Path and file name of the input file
    Returns:
        list: [(user_id, item_id, label), ...]
    """
    score_dict = get_ave_score(input_path_file)
    df = pd.read_csv(input_path_file)
    # print(df.head())
    df.columns = ['user_id', 'item_id', 'rating'] + df.columns[3:].tolist()
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    # print(df.head())

    # Separate into positive and negative interactions
    pos_df = df[df['rating'] >= score_thr].copy()
    neg_df = df[df['rating'] < score_thr].copy()

    # Add labels
    pos_df['label'] = 1
    neg_df['label'] = 0

    # Map negative ratings with item average scores for sampling
    neg_df['item_score'] = neg_df['item_id'].map(score_dict)

    train_data = []

    # Group by user
    pos_group = pos_df.groupby('user_id')
    neg_group = neg_df.groupby('user_id')

    for user_id, pos_items in pos_group:
        neg_items = neg_group.get_group(user_id) if user_id in neg_group.groups else pd.DataFrame()
        data_num = min(len(pos_items), len(neg_items))
        if data_num == 0:
            continue

        # Sample positive interactions
        pos_samples = pos_items[['user_id', 'item_id', 'label']].head(data_num).values.tolist()

        # Sort negative interactions by average item score descending and sample
        neg_sorted = neg_items.sort_values(by='item_score', ascending=False).head(data_num)
        neg_samples = neg_sorted[['user_id', 'item_id']].copy()
        neg_samples['label'] = 0
        neg_samples = neg_samples[['user_id', 'item_id', 'label']].values.tolist()

        train_data.extend(pos_samples)
        train_data.extend(neg_samples)

    return train_data


def lfm_train(train_data, F, alpha, beta, step):
    """
    Args:
        train_data: train_data for lfm
        F: user vector len, item vector len
        alpha:regularization factor
        beta: learning rate
        step: iteration num
    Returns:
        dict: key item_id, value:np.ndarray
        dict: key user_id, value:np.ndarray
    """
    user_vec = {}
    item_vec = {}
    for step_index in range(step):
        random.shuffle(train_data)
        for user_id, item_id, label in train_data:
            if user_id not in user_vec:
                user_vec[user_id] = init_model(F)
            if item_id not in item_vec:
                item_vec[item_id] = init_model(F)

            pred = model_predict(user_vec[user_id], item_vec[item_id])
            delta = label - pred

            user_vec_old = user_vec[user_id].copy()
            item_vec_old = item_vec[item_id].copy()

            # Gradient updates
            user_vec[user_id] += beta * (delta * item_vec_old - alpha * user_vec_old)
            item_vec[item_id] += beta * (delta * user_vec_old - alpha * item_vec_old)
        beta *= 0.9  # Decay learning rate
    return user_vec, item_vec


def init_model(vector_len):
    """
    Args:
        vector_len: the len of vector
    Returns:
         a ndarray
    """
    return np.random.randn(vector_len)


def model_predict(user_vector, item_vector):
    """
    user_vector and item_vector distance
    Args:
        user_vector: model produce user vector
        item_vector: model produce item vector
    Returns:
         a number
    """
    res = np.dot(user_vector, item_vector) / (np.linalg.norm(user_vector) * np.linalg.norm(item_vector))
    return res


def model_train_process():
    """
    test lfm model train
    """
    train_data = get_train_data(local_click_path_file)
    # print(train_data)
    user_vec, item_vec = lfm_train(train_data, 50, 0.01, 0.1, 100)
    for user_id in user_vec:
        recom_result = give_recom_result(user_vec, item_vec, user_id)
        ana_recom_result(train_data, user_id, recom_result)


def give_recom_result(user_vec, item_vec, user_id, fix_num=10):
    """
    Generate top-N recommendation results for a given user using LFM model.

    Args:
        user_vec (dict): user embeddings {user_id: vector}
        item_vec (dict): item embeddings {item_id: vector}
        user_id (str or int): target user ID
        fix_num (int): number of recommendations to return

    Returns:
        list: [(item_id, score), ...] sorted by score descending
    """
    if user_id not in user_vec:
        return []

    user_vector = user_vec[user_id]
    record = {}

    for item_id, item_vector in item_vec.items():
        # Cosine similarity
        denom = np.linalg.norm(user_vector) * np.linalg.norm(item_vector)
        if denom == 0:
            score = 0
        else:
            score = np.dot(user_vector, item_vector) / denom
        record[item_id] = score

    # Sort by score descending and return top-N
    sorted_items = sorted(record.items(), key=operator.itemgetter(1), reverse=True)[:fix_num]
    recom_list = [(item_id, round(score, 3)) for item_id, score in sorted_items]

    return recom_list


def ana_recom_result(train_data, user_id, recom_list):
    """
    Analyze and debug recommendation results for a specific user.

    Args:
        train_data (list): Training data [(user_id, item_id, label), ...]
        user_id (str or int): User ID to analyze
        recom_list (list): Recommendation result from LFM [(item_id, score), ...]
    """
    item_info = get_item_info(local_item_path_file)
    print(f"\n[User {user_id}] Ground Truth Liked Items:")
    for data_instance in train_data:
        tmp_user_id, item_id, label = data_instance
        if tmp_user_id == user_id and label == 1:
            info = item_info.get(item_id, ["Unknown Title"])
            print(f"{item_id}: {info[0]}")

    print(f"\n[User {user_id}] Top-{len(recom_list)} Recommended Items:")
    for item_id, score in recom_list:
        info = item_info.get(item_id, ["Unknown Title"])
        print(f"{item_id}: {info[0]} | Score: {score}")


if __name__ == "__main__":
    # get_item_info(local_item_path_file)
    # print(get_ave_score(local_click_path_file))
    get_train_data(local_click_path_file)
    model_train_process()
