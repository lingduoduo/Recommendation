#-*-coding:utf8-*-
"""
date:2018****
personal rank main algo
"""

import operator
import numpy as np
import pandas as pd
from scipy.sparse.linalg import gmres
from scipy.sparse import coo_matrix, eye
import os
from collections import defaultdict

def get_graph_from_data(input_file):
    """
    Args:
        input_file: user-item-rating CSV file
    Return:
        A dict: {UserA: {itemB: 1, itemC: 1}, itemB: {UserA: 1}, ...}
    """
    if not os.path.exists(input_file):
        return {}

    df = pd.read_csv(input_file)
    score_thr = 4.0

    # Filter and clean
    df = df[df.iloc[:, 2].astype(float) >= score_thr]
    df.columns = ['user', 'item', 'rating', 'timestamp']
    df['item'] = df['item'].astype(str).apply(lambda x: f'item_{x}')

    # Use apply to populate the graph
    graph = defaultdict(dict)
    df.apply(lambda row: (graph[row['user']].update({row['item']: 1}),
                          graph[row['item']].update({row['user']: 1})),
             axis=1)

    return dict(graph)


def get_item_info(input_file):
    """
    Get item info: [title, genre]
    Args:
        input_file: item info file
    Return:
        A dict: key = itemid, value = [title, genre]
    """
    if not os.path.exists(input_file):
        return {}

    # Read the file with no type inference
    df = pd.read_csv(input_file, header=0, dtype=str, keep_default_na=False)

    # Use apply with a lambda to handle different lengths
    item_info = df.apply(
        lambda row: (
            row[0],
            [",".join(row[1:-1]), row[-1]] if len(row) > 3 else [row[1], row[2]]
        ),
        axis=1
    ).to_dict()

    return item_info


def graph_to_m(graph):
    """
    Convert a user-item graph into a sparse matrix (COO format).

    Args:
        graph: dict, user-item graph

    Returns:
        m: scipy.sparse.coo_matrix, normalized transition matrix
        vertex: list of node names
        address_dict: dict mapping node name to row/column index
    """
    # Use pandas Index for fast mapping and alignment
    vertex = pd.Index(graph.keys())
    address_dict = vertex.to_series().to_dict()

    rows = []
    cols = []
    data = []

    for src_node, neighbors in graph.items():
        src_idx = address_dict[src_node]
        weight = round(1 / len(neighbors), 3) if neighbors else 0.0
        for dst_node in neighbors:
            dst_idx = address_dict[dst_node]
            rows.append(src_idx)
            cols.append(dst_idx)
            data.append(weight)

    m = coo_matrix((data, (rows, cols)), shape=(len(vertex), len(vertex)))
    return m, vertex.tolist(), address_dict


def personal_rank(graph, root, alpha, iter_num, recom_num=10):
    """
    Personalized PageRank using pandas where applicable.

    Args:
        graph: dict, user-item bipartite graph
        root: str, the user node to generate recommendations for
        alpha: float, probability of continuing the random walk
        iter_num: int, number of iterations
        recom_num: int, number of recommended items to return

    Returns:
        dict: item_id -> score
    """
    if root not in graph:
        return {}

    # Initialize rank
    rank = pd.Series({node: 0.0 for node in graph})
    rank[root] = 1.0

    for _ in range(iter_num):
        tmp_rank = pd.Series({node: 0.0 for node in graph})
        for out_node, neighbors in graph.items():
            out_degree = len(neighbors)
            if out_degree == 0:
                continue
            for neighbor in neighbors:
                tmp_rank[neighbor] += round(alpha * rank[out_node] / out_degree, 4)
                if neighbor == root:
                    tmp_rank[neighbor] += round(1 - alpha, 4)
        if tmp_rank.equals(rank):
            break
        rank = tmp_rank

    # Filter for item nodes not already connected to the root user
    recom_result = {}
    for node, score in rank.sort_values(ascending=False).items():
        if "_" not in node:
            continue
        if node in graph[root]:
            continue
        recom_result[node] = round(score, 4)
        if len(recom_result) >= recom_num:
            break

    return recom_result


def mat_all_point(m_mat, vertex, alpha):
    """
    Compute E - alpha * M^T

    Args:
        m_mat: scipy.sparse matrix (transition matrix from graph)
        vertex: list of all nodes (users + items)
        alpha: float, probability of continuing random walk

    Returns:
        scipy.sparse.csr_matrix: (E - alpha * M^T)
    """
    total_len = len(vertex)

    # Identity matrix E (sparse)
    identity = eye(total_len, format='csr')

    # Return E - alpha * M.T
    return identity - alpha * m_mat.transpose().tocsr()


def personal_rank_mat(graph, root, alpha, recom_num=10):
    """
    Personalized PageRank using matrix formulation and pandas for score processing.

    Args:
        graph: dict, user-item graph
        root: str, the fixed user node
        alpha: float, random walk probability
        recom_num: int, number of recommendations to return

    Returns:
        dict: item_id -> score
    """
    # Convert graph to matrix, get address_dict and vertex list
    m, vertex, address_dict = graph_to_m(graph)
    if root not in address_dict:
        return {}

    # Build initial vector r0
    index = address_dict[root]
    r0 = np.zeros((len(vertex), 1))
    r0[index][0] = 1.0

    # Build transition matrix and solve A * r = r0 using GMRES
    mat_all = mat_all_point(m, vertex, alpha)
    r, _ = gmres(mat_all, r0.flatten(), tol=1e-8)

    # Convert results to pandas Series
    rank_series = pd.Series(r, index=vertex)

    # Filter to item nodes not already connected to the root
    is_item = rank_series.index.to_series().str.contains("_")
    not_interacted = ~rank_series.index.isin(graph[root])
    filtered = rank_series[is_item & not_interacted]

    # Sort and pick top N
    top_recommendations = filtered.sort_values(ascending=False).head(recom_num)

    # Round and convert to dict
    return top_recommendations.round(3).to_dict()


def get_one_user_recom():
    """
    give one fix_user recom result
    """
    user = "1"
    alpha = 0.8
    graph = get_graph_from_data("../data/ratings.txt")
    iter_num = 100
    recom_result = personal_rank(graph, user, alpha, iter_num, 100)
    return  recom_result
    """
    item_info = read.get_item_info("../data/movies.txt")
    for itemid in graph[user]:
        pure_itemid = itemid.split("_")[1]
        print item_info[pure_itemid]
    print "result---"
    for itemid in recom_result:
        pure_itemid = itemid.split("_")[1]
        print item_info[pure_itemid]
        print recom_result[itemid]
    """


def get_one_user_by_mat():
    """
    give one fix user by mat
    """
    user = "1"
    alpha = 0.8
    graph = get_graph_from_data("../data/ratings.txt")
    recom_result = personal_rank_mat(graph, user, alpha, 100)
    return recom_result


if __name__ == "__main__":
    # print(get_graph_from_data("../data/log.txt"))
    # graph = get_graph_from_data("../data/ratings.txt")
    # print(graph[1])
    # recom_result_base = get_one_user_recom()
    # recom_result_mat = get_one_user_by_mat()

