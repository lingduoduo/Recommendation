#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : linghypshen@gmail.com
@File    : ConentBased.py
"""
import os
import pandas as pd
from pathlib import Path

from sklearn.metrics import class_likelihood_ratios

# Get the project root directory using pathlib
ROOT_DIR = Path.cwd().parent.parent
local_path = ROOT_DIR / "src" / "data" / "input"

local_click_ts_file = "search_click_ts.csv"
local_click_ts_path_file = local_path / local_click_ts_file

local_item_clientid_file = "item_desc_clientid.csv"
local_item_clientid_path_file = local_path / local_item_clientid_file

score_thr = 1


def load_click_data(input_path_file=local_click_ts_file):
    """
    Args:
        input_path_file: user rating file
    Returns:
        dataframe:
        Load CSV with headers skipped manually (first row is header)

    """
    return pd.read_csv(
        input_path_file, 
        skiprows=1, 
        header=None,
        names=["user_id", "item_id", "timestamp", "rating"],
        dtype={"user_id": str, "item_id": str, "timestamp": int, "rating": float}
    ).dropna(subset=["timestamp"])


def get_ave_score(click_df):
    """
    Args:
        click_df: click dataframe
    Returns:
        dict: key = item_id, value = average score (rounded to 3 decimals)
    """
    return click_df.groupby("item_id")["rating"].sum().round(3).to_dict()


def get_latest_timestamp(click_df):
    """
    Args:
        input_path_file: user rating file (columns: userid, item_id, rating, timestamp)
    """
    return click_df["timestamp"].max()


def get_time_score(click_df, latest_ts):
    """
    Args:
        click_df: click dataframe
        timestamp:input timestamp
    Returns:
        time score
    """
    total_sec = 24 * 60 * 60
    delta = (latest_ts - click_df["timestamp"]) / total_sec / 100
    return (1 / (1 + delta)).round(3)


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

    item_cate = (df_expl.groupby("item_id", group_keys=False)
                 .agg({"category": list, "ratio": list})
                 .apply(lambda x: dict(zip(x["category"], x["ratio"])), axis=1)
                 .to_dict())

    top_items = df_expl.sort_values("score", ascending=False)
    cate_item_sort = top_items.groupby("category")["item_id"].apply(lambda x: x.head(topk).tolist()).to_dict()

    return item_cate, cate_item_sort


def get_user_profile(click_df, item_cate, latest_ts):
    """
    Compute user preferences from ratings and item-category mappings.

    Args:
        click_df: click dataframe
        item_cate (dict): {item_id: {cate: ratio}}
        input_path_file (str): Path to user rating CSV file.

    Returns:
        dict: {userid: [(cate1, ratio1), (cate2, ratio2), ...]}
    """
    df = click_df[click_df["rating"] >= score_thr]
    df = df[df["item_id"].isin(item_cate)]
    # Apply time score
    df["time_score"] = get_time_score(df, latest_ts)

    # Build expanded DataFrame
    expanded = []
    for _, row in df.iterrows():
        for cate, ratio in item_cate[row["item_id"]].items():
            score = row["rating"] * row["time_score"] * ratio
            expanded.append((row["user_id"], cate, score))

    df_exp = pd.DataFrame(expanded, columns=["user_id", "cate", "weighted_score"])
    grouped = df_exp.groupby(["user_id", "cate"])["weighted_score"].sum().reset_index()
    return {uid: topk_normalized(group.drop(columns="user_id")) for uid, group in grouped.groupby("user_id")}


def recommend(user_id, user_profile, cate_item_sort, topk=100):
    """
    Args:
        cate_item_sort (dict): {cate: [sorted_item_id1, sorted_item_id2, ...]}
        user_profile (dict): {user_id: [(cate, ratio), ...]}
        user_id (str): user ID to generate recommendations for
        topk (int): number of total recommendations
    Returns:
        dict: {user_id: [item_id1, item_id2, ...]}
    """
    # Build a DataFrame from user"s category interests
    df = pd.DataFrame(user_profile[user_id], columns=["cate", "ratio"])
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
    click_df = load_click_data(local_click_ts_path_file)
    ave_score = get_ave_score(click_df)
    latest_ts = get_latest_timestamp(click_df)
    item_cate, cate_item_sort = get_item_cate(ave_score, local_item_clientid_path_file)
    user_profile = get_user_profile(click_df, item_cate, latest_ts)
    return recommend(user_id, user_profile, cate_item_sort)


if __name__ == "__main__":
    print(run_main("d362d53c-48d8-4537-864d-a4157701a864"))
