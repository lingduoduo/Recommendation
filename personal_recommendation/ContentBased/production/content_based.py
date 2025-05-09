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
        input_file: user rating file
    Return:
        dict: key = itemid, value = average score (rounded to 3 decimals)
    """
    # Load CSV with headers skipped manually (first row is header)
    df = pd.read_csv(input_file, skiprows=1, header=None, names=['userid', 'itemid', 'rating', 'timestamp'])

    # Ensure rating column is float and itemid is str (optional safety)
    df['rating'] = df['rating'].astype(float)
    df['itemid'] = df['itemid'].astype(str)

    # Group by itemid and calculate the mean rating
    ave_score_series = df.groupby('itemid')['rating'].mean().round(3)

    # Convert Series to dict
    return ave_score_series.to_dict()

def get_item_cate(ave_score, input_file, topk=100):
    """
    Args:
        ave_score: dict, key = itemid, value = average rating score
        input_file: item info file (with categories)
    Return:
        item_cate: dict, key = itemid, value = {category: ratio}
        cate_item_sort: dict, key = category, value = top itemids list
    """
    if not os.path.exists(input_file):
        return {}, {}

    # Read file skipping header
    df = pd.read_csv(input_file, skiprows=1, header=None, names=['itemid', 'name', 'categories'])

    # Drop rows with missing values in categories
    df = df.dropna(subset=['categories'])

    # Split categories into lists
    df['cate_list'] = df['categories'].apply(lambda x: x.split('|'))

    # Calculate equal ratio for each category per item
    df['cate_ratio'] = df['cate_list'].apply(lambda x: round(1 / len(x), 3) if len(x) > 0 else 0)

    # Build item_cate: itemid -> {cate: ratio}
    item_cate = {}
    for _, row in df.iterrows():
        itemid = str(row['itemid'])
        ratio = row['cate_ratio']
        item_cate[itemid] = {cate: ratio for cate in row['cate_list']}

    # Build category -> {itemid: score}
    df['itemid'] = df['itemid'].astype(str)
    df['score'] = df['itemid'].apply(lambda x: ave_score.get(x, 0))
    cate_item_map = {}

    for _, row in df.iterrows():
        for cate in row['cate_list']:
            if cate not in cate_item_map:
                cate_item_map[cate] = []
            cate_item_map[cate].append((row['itemid'], row['score']))

    # For each category, sort items by score and keep topk
    cate_item_sort = {
        cate: [itemid for itemid, _ in sorted(items, key=lambda x: x[1], reverse=True)[:topk]]
        for cate, items in cate_item_map.items()
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
    print(len(up))
    print(up["1"])
    return recom(cate_item_sort, up, "1")

if __name__ == "__main__":
    # avg = get_ave_score("../data/ratings.txt")
    # print(avg[31])
    #
    # item_cate, cate_item_sort = get_item_cate(avg, "../data/movies.txt")
    # print(item_cate["1"])
    # print(cate_item_sort["Children"])
    print(run_main())