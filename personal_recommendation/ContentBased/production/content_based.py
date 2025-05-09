import os
import pandas as pd
from pathlib import Path

# Constants and paths
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "src" / "data" / "input"
CLICK_FILE = DATA_DIR / "search_click_ts.csv"
ITEM_FILE = DATA_DIR / "item_desc_clientid.csv"
SCORE_THRESHOLD = 1
SECONDS_IN_DAY = 86400 * 100  # Adjusted for your scale

def load_click_data(path):
    return pd.read_csv(
        path, skiprows=1, header=None,
        names=["user_id", "item_id", "timestamp", "rating"],
        dtype={"user_id": str, "item_id": str, "timestamp": int, "rating": float}
    ).dropna(subset=["timestamp"])

def get_ave_score(click_df):
    return click_df.groupby("item_id")["rating"].sum().round(3).to_dict()

def get_latest_timestamp(click_df):
    return click_df["timestamp"].max()

def compute_time_score(click_df, latest_ts):
    delta = (latest_ts - click_df["timestamp"]) / SECONDS_IN_DAY
    return (1 / (1 + delta)).round(3)

def get_item_cate(ave_score, path, topk=100):
    df = pd.read_csv(path, skiprows=1, header=None,
                     names=["item_id", "title", "categories"],
                     dtype={"item_id": str, "title": str, "categories": str})
    df = df.dropna(subset=["categories"])
    df["cate_list"] = df["categories"].str.split('|')
    df["ratio"] = (1 / df["cate_list"].str.len()).round(3)

    df_exploded = df.explode("cate_list").rename(columns={"cate_list": "category"})
    df_exploded["score"] = df_exploded["item_id"].map(ave_score).fillna(0)

    item_cate = df_exploded.groupby("item_id").apply(
        lambda x: dict(zip(x["category"], x["ratio"]))
    ).to_dict()

    top_items = df_exploded.sort_values("score", ascending=False)
    cate_item_sort = top_items.groupby("category")["item_id"].apply(lambda x: x.head(topk).tolist()).to_dict()

    return item_cate, cate_item_sort

def get_user_profile(click_df, item_cate, latest_ts):
    df = click_df[click_df["rating"] >= SCORE_THRESHOLD]
    df = df[df["item_id"].isin(item_cate)]
    df["time_score"] = compute_time_score(df, latest_ts)

    # Build expanded DataFrame
    expanded = []
    for _, row in df.iterrows():
        for cate, ratio in item_cate[row["item_id"]].items():
            score = row["rating"] * row["time_score"] * ratio
            expanded.append((row["user_id"], cate, score))

    df_exp = pd.DataFrame(expanded, columns=["user_id", "cate", "weighted_score"])
    grouped = df_exp.groupby(["user_id", "cate"])["weighted_score"].sum().reset_index()

    def normalize(group, topk=100):
        group = group.sort_values("weighted_score", ascending=False).head(topk)
        total = group["weighted_score"].sum()
        group["normalized"] = (group["weighted_score"] / total).round(3) if total else 0
        return list(zip(group["cate"], group["normalized"]))

    return {uid: normalize(group.drop(columns="user_id")) for uid, group in grouped.groupby("user_id")}

def recommend(user_id, user_profile, cate_item_sort, topk=100):
    prefs = pd.DataFrame(user_profile.get(user_id, []), columns=["cate", "ratio"])
    prefs["num"] = (prefs["ratio"] * topk).astype(int) + 1
    prefs = prefs[prefs["cate"].isin(cate_item_sort)]

    recs = prefs.apply(lambda r: cate_item_sort[r["cate"]][:r["num"]], axis=1)
    all_items = pd.Series([i for sublist in recs for i in sublist]).drop_duplicates().tolist()
    return {user_id: all_items[:topk]}

def run_main(user_id):
    click_df = load_click_data(CLICK_FILE)
    ave_score = get_ave_score(click_df)
    latest_ts = get_latest_timestamp(click_df)
    item_cate, cate_item_sort = get_item_cate(ave_score, ITEM_FILE)
    user_profile = get_user_profile(click_df, item_cate, latest_ts)
    return recommend(user_id, user_profile, cate_item_sort)

if __name__ == "__main__":
    print(run_main("570ade82-fdef-4be5-9193-7c8869834bef"))
