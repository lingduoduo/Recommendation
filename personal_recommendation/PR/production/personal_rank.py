#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : linghypshen@gmail.com
@File    : PR.py
"""
import os
import numpy as np
import pandas as pd
from scipy.sparse.linalg import gmres
from scipy.sparse import coo_matrix, eye
from collections import defaultdict
from pathlib import Path

# Get the project root directory using pathlib
ROOT_DIR = Path.cwd().parent.parent
local_path = ROOT_DIR / "src" / "data" / "input"

local_click_file = "search_click.csv"
local_click_path_file = local_path / local_click_file

local_item_file = "item_desc.csv"
local_item_path_file = local_path / local_item_file
score_thr = 1


def get_graph_from_data(input_path_file):
    """
    Args:
        input_path_file (str): Path and File name of the input file
    Returns:
        A dict: {UserA: {itemB: 1, itemC: 1}, itemB: {UserA: 1}, ...}
    """
    df = pd.read_csv(input_path_file)

    # Filter and clean
    df = df[df.iloc[:, 2].astype(float) >= score_thr]
    df.columns = ['user', 'item', 'rating'] if len(df.columns) == 3 else ['user', 'item', 'rating', 'timestamp']
    df['item'] = df['item'].astype(str).apply(lambda x: f'item_{x}')

    # Use apply to populate the graph
    graph = defaultdict(dict)
    df.apply(lambda row: (graph[row['user']].update({row['item']: 1}),
                          graph[row['item']].update({row['user']: 1})),
             axis=1)

    return dict(graph)


def get_item_info(input_path_file):
    """
    Get item info: [title]
    Args:
        input_path_file (str): Path and file name of the input file
    Returns:
        A dict: key = itemid, value = [title]
    """
    # Read the file with no type inference
    df = pd.read_csv(input_path_file, header=0, dtype=str, keep_default_na=False)

    # Use apply with a lambda to handle different lengths
    item_info = df.apply(
        lambda row: (
            row[0],
            [",".join(row[1:-1]), row[-1]] if len(row) > 3 else [row[1], row[2]]
        ),
        axis=1
    ).to_dict()

    return item_info


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
    # All unique nodes from keys and values
    all_nodes = set(graph.keys())
    for neighbors in graph.values():
        all_nodes.update(neighbors)

    vertex = pd.Index(sorted(all_nodes))  # consistent order
    address_dict = {node: idx for idx, node in enumerate(vertex)}

    rows, cols, data = [], [], []
    for src_node, neighbors in graph.items():
        if not neighbors:
            continue
        src_idx = address_dict[src_node]
        weight = round(1 / len(neighbors), 3)
        for dst_node in neighbors:
            dst_idx = address_dict[dst_node]
            rows.append(src_idx)
            cols.append(dst_idx)
            data.append(weight)

    m = coo_matrix((data, (rows, cols)), shape=(len(vertex), len(vertex)))
    return m, vertex.tolist(), address_dict


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
    user = "cf056cdb-c14b-4fe7-abb4-ea899db0e992"
    print(f"user: {user}")
    alpha = 0.8
    graph = get_graph_from_data(local_click_path_file)
    iter_num = 100
    recom_result = personal_rank(graph, user, alpha, iter_num, 10)
    return recom_result


def get_one_user_by_mat():
    """
    give one fix user by mat
    """
    user = "cf056cdb-c14b-4fe7-abb4-ea899db0e992"
    print(f"user: {user}")
    alpha = 0.8
    graph = get_graph_from_data(local_click_path_file)
    recom_result = personal_rank_mat(graph, user, alpha, 10)
    return recom_result


if __name__ == "__main__":
    recom_result_base = get_one_user_recom()
    print(recom_result_base)
    print("-------------------------")
    recom_result_mat = get_one_user_by_mat()
    print(recom_result_mat)
