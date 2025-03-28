# -*- coding:utf8 -*-
"""
author: david (converted to PyTorch by ChatGPT)
date: 2025
LFM model train main function with PyTorch
"""

import torch
import torch.nn.functional as F
import sys
import operator
sys.path.append("../util")
import util.read as read


def lfm_train(train_data, F_dim, alpha, beta, step):
    """
    Args:
        train_data: training data for lfm
        F_dim: latent vector dimension
        alpha: regularization factor
        beta: learning rate
        step: number of iterations
    Returns:
        dict of user vectors
        dict of item vectors
    """
    user_vec = {}
    item_vec = {}

    for step_index in range(step):
        for userid, itemid, label in train_data:
            if userid not in user_vec:
                user_vec[userid] = init_model(F_dim)
            if itemid not in item_vec:
                item_vec[itemid] = init_model(F_dim)

            user_vector = user_vec[userid]
            item_vector = item_vec[itemid]

            pred = model_predict(user_vector, item_vector)
            delta = label - pred

            # Gradient update
            user_grad = delta * item_vector - alpha * user_vector
            item_grad = delta * user_vector - alpha * item_vector

            user_vec[userid] += beta * user_grad
            item_vec[itemid] += beta * item_grad

        beta *= 0.9  # Learning rate decay

    return user_vec, item_vec


def init_model(vector_len):
    """
    Returns:
        Randomly initialized PyTorch vector
    """
    return torch.randn(vector_len, dtype=torch.float32)


def model_predict(user_vector, item_vector):
    """
    Cosine similarity between user and item vector
    """
    return F.cosine_similarity(user_vector.unsqueeze(0), item_vector.unsqueeze(0)).item()


def model_train_process():
    train_data = read.get_train_data("../data/ratings.txt")
    user_vec, item_vec = lfm_train(train_data, F_dim=50, alpha=0.01, beta=0.1, step=50)
    for userid in user_vec:
        recom_result = give_recom_result(user_vec, item_vec, userid)
        # ana_recom_result(train_data, userid, recom_result)


def give_recom_result(user_vec, item_vec, userid, fix_num=10):
    """
    Recommend items for given userid
    """
    if userid not in user_vec:
        return []

    user_vector = user_vec[userid]
    scores = {}

    for itemid, item_vector in item_vec.items():
        sim_score = model_predict(user_vector, item_vector)
        scores[itemid] = sim_score

    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    return [(itemid, round(score, 3)) for itemid, score in sorted_scores[:fix_num]]


def ana_recom_result(train_data, userid, recom_list):
    """
    Debug recommendations
    """
    item_info = read.get_item_info("../data/movies.txt")
    for uid, itemid, label in train_data:
        if uid == userid and label == 1:
            print(item_info.get(itemid))
    print("recom result")
    for itemid, score in recom_list:
        print(item_info.get(itemid))


if __name__ == "__main__":
    model_train_process()
