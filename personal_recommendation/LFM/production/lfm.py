#-*-coding:utf8-*-
"""
author:linghypshen@gmail.com
lfm model train main function
"""

import numpy as np
import operator
import os
import pandas as pd
import random

def get_item_info(input_file):
    """
    Load item info using pandas.
    Returns a dictionary: key = itemid, value = [title, genre]
    """
    if not os.path.exists(input_file):
        return {}

    # Read the CSV file; assume the first row is the header
    df = pd.read_csv(input_file, encoding='utf-8')

    # Make sure the file has at least 3 columns
    if df.shape[1] < 3:
        return {}

    # Extract itemid, title, and genre
    # Handle cases where title includes commas by assuming genre is the last column
    df['itemid'] = df.iloc[:, 0]
    df['genre'] = df.iloc[:, -1]
    df['title'] = df.iloc[:, 1:-1].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)

    item_info = dict(zip(df['itemid'], zip(df['title'], df['genre'])))
    return item_info

def get_ave_score(input_file):
    """
    Compute average rating score per item using pandas.
    Args:
        input_file (str): Path to the user rating file
    Returns:
        dict: key = itemid, value = average score (rounded to 3 decimals)
    """
    if not os.path.exists(input_file):
        return {}

    # Load the file with pandas, skip bad lines if any
    df = pd.read_csv(input_file, encoding='utf-8')

    # Ensure there are at least 3 columns: userid, itemid, rating
    if df.shape[1] < 3:
        return {}

    # Rename columns if not already named
    df.columns = ['userid', 'itemid', 'rating'] + df.columns[3:].tolist()

    # Convert ratings to float and group by itemid
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    score_series = df.groupby('itemid')['rating'].mean().round(3)

    return score_series.to_dict()


def get_train_data(input_file):
    """
    Generate training data for LFM model using pandas.
    Args:
        input_file (str): Path to user-item rating CSV file
    Returns:
        list: [(userid, itemid, label), ...]
    """
    if not os.path.exists(input_file):
        return []

    score_dict = get_ave_score(input_file)
    score_thr = 4.0

    df = pd.read_csv(input_file)
    print(df.head())
    df.columns = ['userid', 'itemid', 'rating'] + df.columns[3:].tolist()
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

    # Separate into positive and negative interactions
    pos_df = df[df['rating'] >= score_thr].copy()
    neg_df = df[df['rating'] < score_thr].copy()

    # Add labels
    pos_df['label'] = 1
    neg_df['label'] = 0

    # Map negative ratings with item average scores for sampling
    neg_df['item_score'] = neg_df['itemid'].map(score_dict)

    train_data = []

    # Group by user
    pos_group = pos_df.groupby('userid')
    neg_group = neg_df.groupby('userid')

    for userid, pos_items in pos_group:
        neg_items = neg_group.get_group(userid) if userid in neg_group.groups else pd.DataFrame()
        data_num = min(len(pos_items), len(neg_items))
        if data_num == 0:
            continue

        # Sample positive interactions
        pos_samples = pos_items[['userid', 'itemid', 'label']].head(data_num).values.tolist()

        # Sort negative interactions by average item score descending and sample
        neg_sorted = neg_items.sort_values(by='item_score', ascending=False).head(data_num)
        neg_samples = neg_sorted[['userid', 'itemid']].copy()
        neg_samples['label'] = 0
        neg_samples = neg_samples[['userid', 'itemid', 'label']].values.tolist()

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
    Return:
        dict: key itemid, value:np.ndarray
        dict: key userid, value:np.ndarray
    """
    user_vec = {}
    item_vec = {}
    for step_index in range(step):
        random.shuffle(train_data)
        for userid, itemid, label in train_data:
            if userid not in user_vec:
                user_vec[userid] = init_model(F)
            if itemid not in item_vec:
                item_vec[itemid] = init_model(F)

            pred = model_predict(user_vec[userid], item_vec[itemid])
            delta = label - pred

            user_vec_old = user_vec[userid].copy()
            item_vec_old = item_vec[itemid].copy()

            # Gradient updates
            user_vec[userid] += beta * (delta * item_vec_old - alpha * user_vec_old)
            item_vec[itemid] += beta * (delta * user_vec_old - alpha * item_vec_old)
        beta *= 0.9  # Decay learning rate
    return user_vec, item_vec


def init_model(vector_len):
    """
    Args:
        vector_len: the len of vector
    Return:
         a ndarray
    """
    return np.random.randn(vector_len)


def model_predict(user_vector, item_vector):
    """
    user_vector and item_vector distance
    Args:
        user_vector: model produce user vector
        item_vector: model produce item vector
    Return:
         a num
    """
    res = np.dot(user_vector, item_vector)/(np.linalg.norm(user_vector)*np.linalg.norm(item_vector))
    return res


def model_train_process():
    """
    test lfm model train
    """
    train_data = get_train_data("../data/ratings.txt")
    user_vec, item_vec = lfm_train(train_data, 50, 0.01, 0.1, 50)
    for userid in user_vec:
        recom_result = give_recom_result(user_vec, item_vec, userid)
        ana_recom_result(train_data, userid, recom_result)

def give_recom_result(user_vec, item_vec, userid, fix_num=10):
    """
    Generate top-N recommendation results for a given user using LFM model.

    Args:
        user_vec (dict): user embeddings {userid: vector}
        item_vec (dict): item embeddings {itemid: vector}
        userid (str or int): target user ID
        fix_num (int): number of recommendations to return

    Returns:
        list: [(itemid, score), ...] sorted by score descending
    """
    if userid not in user_vec:
        return []

    user_vector = user_vec[userid]
    record = {}

    for itemid, item_vector in item_vec.items():
        # Cosine similarity
        denom = np.linalg.norm(user_vector) * np.linalg.norm(item_vector)
        if denom == 0:
            score = 0
        else:
            score = np.dot(user_vector, item_vector) / denom
        record[itemid] = score

    # Sort by score descending and return top-N
    sorted_items = sorted(record.items(), key=operator.itemgetter(1), reverse=True)[:fix_num]
    recom_list = [(itemid, round(score, 3)) for itemid, score in sorted_items]

    return recom_list

def ana_recom_result(train_data, userid, recom_list):
    """
    Analyze and debug recommendation results for a specific user.

    Args:
        train_data (list): Training data [(userid, itemid, label), ...]
        userid (str or int): User ID to analyze
        recom_list (list): Recommendation result from LFM [(itemid, score), ...]
    """
    item_info = get_item_info("../data/movies.txt")

    print(f"\n[User {userid}] Ground Truth Liked Items:")
    for data_instance in train_data:
        tmp_userid, itemid, label = data_instance
        if tmp_userid == userid and label == 1:
            info = item_info.get(itemid, ["Unknown Title", "Unknown Genre"])
            print(f"{itemid}: {info[0]} ({info[1]})")

    print(f"\n[User {userid}] Top-{len(recom_list)} Recommended Items:")
    for itemid, score in recom_list:
        info = item_info.get(itemid, ["Unknown Title", "Unknown Genre"])
        print(f"{itemid}: {info[0]} ({info[1]}) | Score: {score}")


if __name__ == "__main__":
    # train_data = get_train_data("../data/ratings.txt")
    # print(len(train_data))

    model_train_process()
