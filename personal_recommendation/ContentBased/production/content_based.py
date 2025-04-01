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
        ave_score (dict): {itemid: rating score}
        input_file (str): item info file with columns [itemid,...,cate_str]

    Returns:
        item_cate (dict): {itemid: {cate: ratio}}
        cate_item_sort (dict): {cate: [itemid1, itemid2, ...]}
    """
    if not os.path.exists(input_file):
        return {}, {}

    # Load item info file, skip header
    df = pd.read_csv(input_file, skiprows=1, header=None)

    # Assume itemid is column 0, and categories are in the last column
    df = df[[0, df.columns[-1]]]
    df.columns = ["itemid", "cate_str"]
    df.dropna(subset=["itemid", "cate_str"], inplace=True)
    df["itemid"] = df["itemid"].astype(str)

    # Split cate_str into list and compute ratio per item
    df["cate_list"] = df["cate_str"].apply(lambda x: x.strip().split("|"))
    df["cate_ratio_dict"] = df["cate_list"].apply(
        lambda cates: {cate: round(1 / len(cates), 3) for cate in cates}
    )

    # -------- item_cate: {itemid: {cate: ratio}} --------
    item_cate = pd.Series(df["cate_ratio_dict"].values, index=df["itemid"]).to_dict()

    # -------- record: {cate: {itemid: score}} --------
    # Explode for category-level mapping
    df_exploded = df[["itemid", "cate_list"]].explode("cate_list")
    df_exploded.columns = ["itemid", "cate"]
    df_exploded["score"] = df_exploded["itemid"].apply(lambda x: ave_score.get(x, 0))

    # Pivot into nested dict: {cate: {itemid: score}}
    record = (
        df_exploded.groupby("cate")[["itemid", "score"]]
        .apply(lambda x: dict(zip(x["itemid"], x["score"])))
        .to_dict()
    )

    # -------- cate_item_sort: {cate: [itemid1, itemid2, ...]} --------
    cate_item_sort = {
        cate: [
            itemid for itemid, _ in sorted(items.items(), key=lambda x: x[1], reverse=True)[:topk]
        ]
        for cate, items in record.items()
    }

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


def get_up(item_cate, input_file, score_thr=4.0, topk=2):
    """
    Args:
        item_cate (dict): {itemid: {cate: ratio}}
        input_file (str): user rating file
    Returns:
        dict: {userid: [(cate1, ratio1), (cate2, ratio2), ...]}
    """
    if not os.path.exists(input_file):
        return {}

    # Load rating file
    df = pd.read_csv(input_file, skiprows=1, header=None, names=["userid", "itemid", "rating", "timestamp"])

    # Filter: valid ratings and known items
    df = df[df["rating"] >= score_thr]
    df = df[df["itemid"].isin(item_cate)]

    # Add time_score column using your get_time_score() function
    df["time_score"] = df["timestamp"].apply(get_time_score)

    # Expand each itemid → category mapping into multiple rows
    df["cate_ratios"] = df["itemid"].map(item_cate)
    df = df.explode("cate_ratios")  # This gives key-value dict per row

    # Flatten nested cate:ratio dicts into two columns
    df = df[df["cate_ratios"].notnull()]
    df["cate"] = df["cate_ratios"].apply(lambda x: list(x.keys())[0])
    df["ratio"] = df["cate_ratios"].apply(lambda x: list(x.values())[0])

    # Compute weighted score: rating * time_score * ratio
    df["weighted_score"] = df["rating"] * df["time_score"] * df["ratio"]

    # Aggregate: sum scores by user and category
    agg_df = df.groupby(["userid", "cate"])["weighted_score"].sum().reset_index()

    # Rank and normalize top-k categories per user
    def topk_normalized(group):
        top = group.sort_values("weighted_score", ascending=False).head(topk)
        total = top["weighted_score"].sum()
        return list(zip(top["cate"], (top["weighted_score"] / total).round(3)))

    up = agg_df.groupby("userid").apply(topk_normalized).to_dict()

    return up


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
    if userid not in up:
        return {}

    # Build a DataFrame from user's category interests
    df = pd.DataFrame(up[userid], columns=["cate", "ratio"])

    # Calculate number of items to recommend from each category
    df["num"] = df["ratio"].apply(lambda r: int(topk * r) + 1)

    # Filter to only available categories in cate_item_sort
    df = df[df["cate"].isin(cate_item_sort)]

    # For each category, take the top N items
    df["items"] = df.apply(lambda row: cate_item_sort[row["cate"]][:row["num"]], axis=1)

    # Flatten and deduplicate the final recommendation list
    all_items = df["items"]._


def run_main():
    ave_score = get_ave_score("../data/ratings.txt")
    item_cate, cate_item_sort = get_item_cate(ave_score, "../data/movies.txt")
    up = get_up(item_cate, "../data/ratings.txt")
    recom(cate_item_sort, up, "1")

if __name__ == "__main__":
    avg = get_ave_score("../data/ratings.txt")
    print(avg[31])

    item_cate, cate_item_sort = get_item_cate(avg, "../data/movies.txt")
    print(item_cate["1"])
    print(cate_item_sort["Children"])
    # run_main()