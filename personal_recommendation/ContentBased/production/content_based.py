#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : ling.huang@adp.com
@File    : ConentBased.py
"""
import os
import pandas as pd
from pathlib import Path

# Get the project root directory using pathlib
ROOT_DIR = Path(__file__).parent.parent.parent
local_path = ROOT_DIR / "src" / "data" / "input"

local_click_ts_file = "search_click_ts.csv"
local_click_ts_path_file = os.path.join(local_path, local_click_ts_file)

local_item_clientid_file = "item_desc_clientid.csv"
local_item_clientid_path_file = os.path.join(local_path, local_item_clientid_file)

score_thr = 1


def get_ave_score(input_path_file):
    """
    Args:
        input_path_file: user rating file
    Returns:
        dict: key = item_id, value = average score (rounded to 3 decimals)
    """
    # Load CSV with headers skipped manually (first row is header)
    df = pd.read_csv(input_path_file,
                     skiprows=1,
                     header=None,
                     names=["user_id", "item_id", "timestamp", "rating"],
                     dtype={"user_id": str, "item_id": str, "timestamp": int, "rating": float},
                     )

    # Group by item_id and calculate the mean rating
    ave_score_series = df.groupby("item_id")["rating"].sum().round(3)

    # Convert Series to dict
    return ave_score_series.to_dict()


def get_latest_timestamp(input_path_file):
    """
    Args:
        input_path_file: user rating file (columns: userid, item_id, rating, timestamp)
    """
    # Load file, skip header
    df = pd.read_csv(input_path_file,
                     skiprows=1,
                     header=None,
                     names=["user_id", "item_id", "timestamp", "rating"],
                     dtype={"user_id": str, "item_id": str, "timestamp": int, "rating": float},
                     )

    # Drop rows with missing timestamp, convert to int
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)

    # Get latest timestamp
    latest = df["timestamp"].max()
    return latest


def topk_normalized(group, topk=100):
    """
    Rank and normalize top-k categories for a user.
    Args:
        group (DataFrame): Rows corresponding to a single user's category scores.
        topk (int): Number of top categories to return.
    Returns:
        list of tuples: [(category, normalized_score), ...]
    """
    top = group.sort_values("weighted_score", ascending=False).head(topk)
    total = top["weighted_score"].sum()
    if total == 0:
        return list(zip(top["cate"], [0] * len(top)))
    return list(zip(top["cate"], (top["weighted_score"] / total).round(3)))


def get_time_score(timestamp):
    """
    Args:
        timestamp:input timestamp
    Returns:
        time score
    """
    fix_time_stamp = get_latest_timestamp(local_click_ts_path_file)
    total_sec = 24 * 60 * 60
    delta = (fix_time_stamp - timestamp) / total_sec / 100
    return round(1 / (1 + delta), 3)

def get_item_cate(ave_score, input_path_file, topk=100):
    """
    Args:
        ave_score: dict, key = item_id, value = average rating score
        input_path_file: item info file (with categories)
    Returns:
        item_cate: dict, key = item_id, value = {category: ratio}
        cate_item_sort: dict, key = category, value = top item_ids list
    """
    # Read file skipping header
    df = pd.read_csv(input_path_file,
                     skiprows=1,
                     header=None,
                     names=["item_id", "title", "categories"],
                     dtype={"item_id": str, "title": str, "categories": str},
                     )
    df = df.dropna(subset=['categories'])
    df['cate_list'] = df['categories'].str.split('|')
    df['ratio'] = (1 / df['cate_list'].str.len()).round(3)

    df_expl = (
        df[['item_id', 'cate_list', 'ratio']]
        .explode('cate_list')
        .rename(columns={'cate_list': 'category'})
    )
    df_expl['score'] = df_expl['item_id'].map(ave_score).fillna(0)

    item_cate = {
        item: grp.set_index('category')['ratio'].to_dict()
        for item, grp in df_expl.groupby('item_id')
    }

    topk_df = (
        df_expl
        .sort_values('score', ascending=False)
        .groupby('category', group_keys=False)
        .head(topk)
    )
    cate_item_sort = (
        topk_df
        .groupby('category')['item_id']
        .apply(list)
        .to_dict()
    )

    return item_cate, cate_item_sort


def get_up(item_cate, input_path_file):
    """
    Compute user preferences from ratings and item-category mappings.

    Args:
        item_cate (dict): {item_id: {cate: ratio}}
        input_path_file (str): Path to user rating CSV file.

    Returns:
        dict: {userid: [(cate1, ratio1), (cate2, ratio2), ...]}
    """
    # Load and filter ratings
    df = pd.read_csv(input_path_file,
                     skiprows=1,
                     header=None,
                     names=["user_id", "item_id", "timestamp", "rating"],
                     dtype={"user_id": str, "item_id": str, "timestamp": int, "rating": float},
                     )

    df = df[df["rating"] >= score_thr]
    df = df[df["item_id"].isin(item_cate)]
    # Apply time score
    df["time_score"] = df["timestamp"].apply(get_time_score)

    # Expand item-cate mapping into rows
    expanded_rows = []
    for _, row in df.iterrows():
        item_id = row["item_id"]
        user_id = row["user_id"]
        rating = row["rating"]
        time_score = row["time_score"]
        for cate, ratio in item_cate[item_id].items():
            weighted = rating * time_score * ratio
            expanded_rows.append((user_id, cate, weighted))
    expanded_df = pd.DataFrame(expanded_rows, columns=["user_id", "cate", "weighted_score"])

    # Aggregate by user-category
    agg_df = expanded_df.groupby(["user_id", "cate"])["weighted_score"].sum().reset_index()

    return {
        user_id: topk_normalized(group.drop(columns="user_id"))
        for user_id, group in agg_df.groupby("user_id")
    }


def recom(cate_item_sort, up, user_id, topk=100):
    """
    Args:
        cate_item_sort (dict): {cate: [sorted_item_id1, sorted_item_id2, ...]}
        up (dict): {user_id: [(cate, ratio), ...]}
        user_id (str): user ID to generate recommendations for
        topk (int): number of total recommendations
    Returns:
        dict: {user_id: [item_id1, item_id2, ...]}
    """
    # Build a DataFrame from user"s category interests
    df = pd.DataFrame(up[user_id], columns=["cate", "ratio"])
    # Calculate number of items to recommend from each category
    df["num"] = df["ratio"].apply(lambda r: int(topk * r) + 1)

    # Filter to only available categories in cate_item_sort
    df = df[df["cate"].isin(cate_item_sort)]

    # For each category, take the top N items
    df["items"] = df.apply(lambda row: cate_item_sort[row["cate"]][:row["num"]], axis=1)

    # Flatten and deduplicate the final recommendation list
    all_items = df["items"].explode().drop_duplicates().tolist()
    return {user_id: all_items[:topk]}  # limit final list to topk items


def run_main(user_id):
    ave_score = get_ave_score(local_click_ts_path_file)
    item_cate, cate_item_sort = get_item_cate(ave_score, local_item_clientid_path_file)
    up = get_up(item_cate, local_click_ts_path_file)
    return recom(cate_item_sort, up, user_id)


if __name__ == "__main__":
    avg = get_ave_score(local_click_ts_path_file)
    item_cate, cate_item_sort = get_item_cate(avg, local_item_clientid_path_file)
    print(cate_item_sort["002"])
    print("===================================================")
    print(run_main("570ade82-fdef-4be5-9193-7c8869834bef"))
