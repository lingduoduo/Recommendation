#-*-coding:utf8-*-
"""
author: linghypshen@gmail.com
get up and online recommendation
"""

import os
import pandas as pd

def get_ave_score(input_file):
    """
    Args:
        input_file: user rating CSV file (columns: userid, itemid, rating, timestamp)

    Returns:
        dict: {itemid: average_rating}
    """
    if not os.path.exists(input_file):
        return {}

    # Load file, skip first line (header), assume comma-separated
    df = pd.read_csv(input_file, skiprows=1, header=None, names=["userid", "itemid", "rating", "timestamp"])

    # Drop rows with missing data
    df = df.dropna(subset=["itemid", "rating"])

    # Ensure proper types
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    # Group by itemid and calculate average rating
    ave_scores = df.groupby("itemid")["rating"].mean().round(3)

    # Convert to dict
    return ave_scores.to_dict()

def get_item_cate(ave_score, input_file, topk=100):
    """
    Args:
        ave_score (dict): {itemid: avg_rating}
        input_file (str): item info file (columns: itemid,...,categories pipe-separated)
        topk (int): number of top items per category to return

    Returns:
        item_cate (dict): {itemid: {cate: ratio}}
        cate_item_sort (dict): {cate: [itemid1, itemid2, ...]}
    """
    # Load item info file, skip header
    df = pd.read_csv(input_file, skiprows=1, header=None)

    # Assume first column is itemid, last is pipe-separated category string
    df = df[[0, df.columns[-1]]]
    df.columns = ["itemid", "cate_str"]

    # Drop missing and ensure string
    df = df.dropna(subset=["itemid", "cate_str"])
    df["itemid"] = df["itemid"].astype(str)
    df["cate_list"] = df["cate_str"].apply(lambda x: x.strip().split("|"))

    # Build item_cate dict
    df["cate_ratio_dict"] = df["cate_list"].apply(
        lambda cates: {cate: round(1 / len(cates), 3) for cate in cates}
    )
    item_cate = pd.Series(df["cate_ratio_dict"].values, index=df["itemid"]).to_dict()

    # Explode for category-level processing
    df_exploded = df[["itemid", "cate_list"]].explode("cate_list")
    df_exploded.columns = ["itemid", "cate"]

    # Add average score from ave_score dict
    df_exploded["score"] = df_exploded["itemid"].map(ave_score).fillna(0.0)

    # Properly sort and group
    df_exploded = df_exploded.sort_values(by=["cate", "score"], ascending=[True, False])
    df_exploded["rank"] = df_exploded.groupby("cate").cumcount()

    # Keep only topk
    df_topk = df_exploded[df_exploded["rank"] < topk]

    # Build cate_item_sort dict
    cate_item_sort = df_topk.groupby("cate")["itemid"].apply(list).to_dict()

    return item_cate, cate_item_sort

def get_latest_timestamp(input_file):
    """
    Args:
        input_file: user rating file (columns: userid, itemid, rating, timestamp)
    """
    if not os.path.exists(input_file):
        return

    # Load file, skip header
    df = pd.read_csv(input_file, skiprows=1, header=None, names=["userid", "itemid", "rating", "timestamp"])

    # Drop rows with missing timestamp, convert to int
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce").fillna(0).astype(int)

    # Get latest timestamp
    latest = df["timestamp"].max()
    print(latest)

def topk_normalized(group, topk=2):
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


def get_up(item_cate, input_file, score_thr=4.0, topk=2):
    """
    Compute user preferences from ratings and item-category mappings.

    Args:
        item_cate (dict): {itemid: {cate: ratio}}
        input_file (str): Path to user rating CSV file.
        score_thr (float): Minimum score to consider a rating.
        topk (int): Number of top categories to return per user.

    Returns:
        dict: {userid: [(cate1, ratio1), (cate2, ratio2), ...]}
    """
    # Load and filter ratings
    df = pd.read_csv(input_file, skiprows=1, header=None, names=["userid", "itemid", "rating", "timestamp"])
    df["itemid"] = df["itemid"].astype(str)
    df["userid"] = df["userid"].astype(str)
    df = df[df["rating"] >= score_thr]

    df = df[df["itemid"].isin(item_cate)]
    # Apply time score
    df["time_score"] = df["timestamp"].apply(get_time_score)

    # Expand item-cate mapping into rows
    expanded_rows = []
    for _, row in df.iterrows():
        itemid = row["itemid"]
        userid = row["userid"]
        rating = row["rating"]
        time_score = row["time_score"]
        for cate, ratio in item_cate[itemid].items():
            weighted = rating * time_score * ratio
            expanded_rows.append((userid, cate, weighted))

    expanded_df = pd.DataFrame(expanded_rows, columns=["userid", "cate", "weighted_score"])

    # Aggregate by user-category
    agg_df = expanded_df.groupby(["userid", "cate"])["weighted_score"].sum().reset_index()

    return {
        userid: topk_normalized(group.drop(columns="userid"))
        for userid, group in agg_df.groupby("userid")
    }


def get_time_score(timestamp):
    """
    Args:
        timestamp:input timestamp
    Return:
        time score
    """
    fix_time_stamp = 1476086345
    total_sec = 24*60*60
    delta = (fix_time_stamp - timestamp)/total_sec/100
    return round(1/(1+delta), 3)


def recom(cate_item_sort, up, userid, topk=10):
    """
    Args:
        cate_item_sort (dict): {cate: [sorted_itemid1, sorted_itemid2, ...]}
        up (dict): {userid: [(cate, ratio), ...]}
        userid (str): user ID to generate recommendations for
        topk (int): number of total recommendations

    Returns:
        dict: {userid: [itemid1, itemid2, ...]}
    """
    # Build a DataFrame from user's category interests
    df = pd.DataFrame(up[userid], columns=["cate", "ratio"])
    # Calculate number of items to recommend from each category
    df["num"] = df["ratio"].apply(lambda r: int(topk * r) + 1)

    # Filter to only available categories in cate_item_sort
    df = df[df["cate"].isin(cate_item_sort)]

    # For each category, take the top N items
    df["items"] = df.apply(lambda row: cate_item_sort[row["cate"]][:row["num"]], axis=1)

    # Flatten and deduplicate the final recommendation list
    all_items = df["items"].explode().drop_duplicates().tolist()

    return {userid: all_items[:topk]}  # limit final list to topk items


def run_main():
    ave_score = get_ave_score("../data/ratings.txt")
    item_cate, cate_item_sort = get_item_cate(ave_score, "../data/movies.txt")
    up = get_up(item_cate, "../data/ratings.txt")
    # print(up)
    return recom(cate_item_sort, up, "1")

if __name__ == "__main__":
    # avg = get_ave_score("../data/ratings.txt")
    # print(avg[31])
    #
    # item_cate, cate_item_sort = get_item_cate(avg, "../data/movies.txt")
    # print(item_cate["1"])
    # print(cate_item_sort["Children"])
    print(run_main())