#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : linghypshen@gmail.com
@File    : search_with_click.py
"""
import os
import pandas as pd
from pathlib import Path

# Get the project root directory using pathlib
ROOT_DIR = Path.cwd().parent.parent
local_path = ROOT_DIR / "src" / "data" / "input"

local_click_file = "search_click.csv"
local_item_file = "item_desc.csv"
local_click_ts_file = "search_click_ts.csv"
local_item_clientid_file = "item_desc_clientid.csv"
local_view_click_file = "view_click.csv"
local_user_desc_file = "user_desc.csv"

# Collect search and click data for the last 7 days
def query_search_with_click(output_path=local_path, output_file=local_click_file):
    """
    Export search and click data for the last `date_range` days to a CSV file.

    Args:
        output_path (str): The file path to save the CSV.
        date_range (int): The number of days to look back for data.
    """
    # Query the data
    query = f"""
        SELECT _token_associate_id AS user_id, 
               click_object_id AS item_id, 
               SUM(click) AS rating
        FROM onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
        WHERE click_object_id IS NOT NULL
        AND action = "actions"
        GROUP BY 1, 2
    """
    df = spark.sql(query)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / output_file
    df.toPandas().to_csv(output_file_path, index=False)

# Collect click label data 
def query_click_description(output_path=local_path, output_file=local_item_file):
    """
    Export click item data to a CSV file.

    Args:
        output_path (str): The file path to save the CSV.
    """
    # SQL query to fetch click item descriptions
    query = """
        SELECT click_object_id AS item_id, 
               click_details_caption AS title
        FROM onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
        WHERE click_object_id IS NOT NULL
        AND action = "actions"
        GROUP BY click_object_id, click_details_caption
    """
    # Execute the query and save the result to a CSV file
    df = spark.sql(query)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / output_file
    df.toPandas().to_csv(output_file_path, index=False)

def query_search_with_click_ts(output_path=local_path, output_file=local_click_ts_file):
    """
    Export click item data to a CSV file.

    Args:
        output_path (str): The file path to save the CSV.
    """
    # SQL query to fetch click item descriptions
    query = """
        SELECT _token_associate_id AS user_id,
                click_object_id AS item_id,
                to_unix_timestamp(time_stamp, "yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'") AS unix_time_stamp,
                SUM(click) AS rating
            FROM onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
            WHERE click_object_id IS NOT NULL
            AND action = "actions"
            GROUP BY 1, 2, 3;
    """
    # Execute the query and save the result to a CSV file
    df = spark.sql(query)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / output_file
    df.toPandas().to_csv(output_file_path, index=False)

def query_click_description_clientid(output_path=local_path, output_file=local_item_clientid_file):
    """
    Export click item data to a CSV file.

    Args:
        output_path (str): The file path to save the CSV.
    """
    # SQL query to fetch click item descriptions
    query = """
        SELECT click_object_id AS item_id, 
               click_details_caption AS title,
               concat_ws('|', collect_set(client_id)) AS categories
        FROM onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
        WHERE click_object_id IS NOT NULL
        AND action = "actions"
        GROUP BY click_object_id, click_details_caption
    """
    # Execute the query and save the result to a CSV file
    df = spark.sql(query)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / output_file
    df.toPandas().to_csv(output_file_path, index=False)

def query_view_click(output_path=local_path, output_file=local_view_click_file):
    """
    Export view and click data to a CSV file.

    Args:
        output_path (str): The directory path to save the CSV file.
        output_file (str): The name of the CSV file.
    """
    # SQL query to fetch view and click data
    query = """
        SELECT
            view._token_associate_id AS user_id,
            view._id AS item_id,
            view.click AS rating
        FROM
            onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click AS view
        JOIN
            (
                SELECT
                    traceId,
                    MAX(resPos) AS max_resPos
                FROM
                    onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
                WHERE
                    click_object_id IS NOT NULL
                GROUP BY
                    traceId
            ) AS click
        ON
            view.traceId = click.traceId
            AND view.resPos <= click.max_resPos
        WHERE view.action = "actions"
    """
    # Execute the query
    df = spark.sql(query)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / output_file
    df.toPandas().to_csv(output_file_path, index=False)
    
def query_user_desc(output_path=local_path, output_file=local_user_desc_file):
    """
    Export view and click data to a CSV file.

    Args:
        output_path (str): The directory path to save the CSV file.
        output_file (str): The name of the CSV file.
    """
    # SQL query to fetch view and click data
    query = """
        SELECT DISTINCT
            _token_associate_id as user_id,
            LAST_VALUE(user_agent) OVER (PARTITION BY _token_associate_id ORDER BY time_stamp ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS last_user_agent
        FROM onedata_us_east_1_shared_dit.nas_raw_lyric_search_dit.ml_search_with_click
    """
    # Execute the query
    df = spark.sql(query)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file_path = output_path / output_file
    df.toPandas().to_csv(output_file_path, index=False)

def load_click_data(input_path=local_path, input_file=local_click_file):
    """
    Load click data from a CSV file.

    Args:
        input_path (str): The file path to load the CSV from.

    Returns:
        pandas.DataFrame: The loaded click data.
    """
    input_file_path = input_path / input_file
    df = pd.read_csv(input_file_path,
                       skiprows=1,
                       names=["user_id", "item_id", "rating"],
                       dtype={"user_id": str, "item_id": str, "rating": str}
                       )
    df["rating"] = pd.to_numeric(df['rating'], errors='coerce').fillna(0).astype(int)
    print("Total count of triplet of user, item, click data is:", len(df))
    MAX_USERS = len(df['user_id'].unique())
    print("Total unique count of users is:", MAX_USERS)
    print("Total unique count of items is:", len(df['item_id'].unique()))
    print("Total count of clicks is:", sum(df['rating']))
    print("---------------------------------------------------------------------------")
    return df

def load_item_data(input_path=local_path, input_file=local_item_file):
    """
    Load item data from a CSV file.

    Args:
        input_path (str): The file path to load the CSV from.

    Returns:
        pandas.DataFrame: The loaded item data.
    """
    input_file_path = input_path / input_file
    df = pd.read_csv(input_file_path,
                       skiprows=1,
                       names=["item_id", "title"],
                       dtype={"item_id": str, "title": str}
                       )
    print("Total item count is: ", len(df))
    print("---------------------------------------------------------------------------")
    return df

def load_click_ts_data(input_path=local_path, input_file=local_item_file):
    """
    Load item data from a CSV file.

    Args:
        input_path (str): The file path to load the CSV from.

    Returns:
        pandas.DataFrame: The loaded item data.
    """
    input_file_path = input_path / input_file
    df = pd.read_csv(input_file_path,
                       skiprows=1,
                       names=["user_id", "item_id", "timestamp", "rating"],
                       dtype={"user_id": str, "item_id": str, "timestamp": int, "rating": str}
                       )
    print("Total click count w.r.t timestamp is: ", len(df))
    print("---------------------------------------------------------------------------")
    return df

def load_click_description_clientid(input_path=local_path, input_file=local_item_clientid_file):
    """
    Load item data from a CSV file.

    Args:
        input_path (str): The file path to load the CSV from.

    Returns:
        pandas.DataFrame: The loaded item data.
    """
    input_file_path = input_path / input_file
    df = pd.read_csv(input_file_path,
                       skiprows=1,
                       names=["item_id", "title", "client_id"],
                       dtype={"item_id": str, "title": str, "client_id": str}
                       )
    print("Total item count given client_id is: ", len(df))
    print("---------------------------------------------------------------------------")
    return df

def load_view_click(input_path=local_path, input_file=local_view_click_file):
    """
    Load item data from a CSV file.

    Args:
        input_path (str): The file path to load the CSV from.

    Returns:
        pandas.DataFrame: The loaded item data.
    """
    input_file_path = input_path / input_file
    df = pd.read_csv(input_file_path,
                       skiprows=1,
                       names=["user_id", "item_id", "rating"],
                       dtype={"user_id": str, "item_id": str, "rating": float}
                       )
    print("Total view count is: ", len(df))
    print("---------------------------------------------------------------------------")
    return df

def load_user_desc(input_path=local_path, input_file=local_user_desc_file):
    """
    Load item data from a CSV file.

    Args:
        input_path (str): The file path to load the CSV from.

    Returns:
        pandas.DataFrame: The loaded item data.
    """
    input_file_path = input_path / input_file
    df = pd.read_csv(input_file_path,
                       skiprows=1,
                       names=["user_id", "last_user_agent"],
                       dtype={"user_id": str, "last_user_agent": str}
                       )
    print("Total user count is: ", len(df))
    print("---------------------------------------------------------------------------")
    return df

if __name__ == "__main__":
    query_search_with_click(output_path=local_path, output_file=local_click_file)
    query_click_description(output_path=local_path, output_file=local_item_file)
    query_search_with_click_ts(output_path=local_path, output_file=local_click_ts_file)
    query_click_description_clientid(output_path=local_path, output_file=local_item_clientid_file)
    query_view_click(output_path=local_path, output_file=local_view_click_file)
    query_user_desc(output_path=local_path, output_file=local_user_desc_file)

    df_clicks = load_click_data(input_path=local_path, input_file=local_click_file)
    print(df_clicks.head())
    print("---------------------------------------------------------------------------")
    
    df_items = load_item_data(input_path=local_path, input_file=local_item_file)
    print(df_items.head())
    print("---------------------------------------------------------------------------")
    
    df_clicks_ts = load_click_ts_data(input_path=local_path, input_file=local_click_ts_file)
    print(df_clicks_ts.head())
    print("---------------------------------------------------------------------------")
    
    df_item_clientid = load_click_description_clientid(input_path=local_path, input_file=local_item_clientid_file)
    print(df_item_clientid.head())
    print("---------------------------------------------------------------------------")
    
    df_view_click = load_view_click(input_path=local_path, input_file=local_view_click_file)
    print(df_view_click.head())
    print("---------------------------------------------------------------------------")
    
    df_user_desc = load_user_desc(input_path=local_path, input_file=local_user_desc_file)
    print(df_user_desc.head())
