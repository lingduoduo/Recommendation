#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : ling.huang@adp.com
@File    : content_based.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path

# --- Load data ---
ROOT_DIR = Path.cwd().parent.parent
local_output_path = ROOT_DIR / "src" / "data" / "output"
source_file = "content_based.csv"
local_file = local_output_path / source_file
output_df = pd.read_csv(local_file)
local_input_path = ROOT_DIR / "src" / "data" / "input"
item_file = "item_desc.csv"
local_file = local_input_path / item_file
item_df = pd.read_csv(local_file)
df = output_df.merge(item_df, on="item_id", how="left")
df.columns = ['user_id', 'click_object_id', 'caption']

# --- Dummy function to simulate recommendation ---
def get_recommendations(user_id, top_k=20):
    # Simple filter: title or description contains the query (case-insensitive)
    filtered = df[df["user_id"] == user_id].drop_duplicates()
    return filtered[:top_k] if len(filtered) >= top_k else filtered# top K results

# --- Streamlit UI ---
st.title("Content Based Recommendation Demo")

user_id = st.text_input("Enter your search user_id:", placeholder="d362d53c-48d8-4537-864d-a4157701a864")
if user_id:
    st.write(f"Showing recommendations for: **{user_id}**")

    results = get_recommendations(user_id)

    if len(results) > 0:
        st.dataframe(results)
    else:
        st.info("No matching results found.")
