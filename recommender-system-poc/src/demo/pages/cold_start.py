# -*- coding: utf-8 -*-
"""
@Author  : ling.huang@adp.com
@File    : cold_start.py
"""
import streamlit as st
import pandas as pd
from pathlib import Path

# --- Load data ---
ROOT_DIR = Path.cwd().parent.parent
local_path = ROOT_DIR / "src" / "data" / "output"
source_file = "cold_start.csv"
local_file = local_path / source_file
df = pd.read_csv(local_file)

# --- Dummy function to simulate recommendation ---
def get_recommendations(client_id, top_k=20):
    # Simple filter: title or description contains the query (case-insensitive)
    filtered = df[df["client_id"] == client_id]
    return filtered[:top_k] if len(filtered) >= top_k else filtered# top K results

# --- Streamlit UI ---
st.title("Cold Start Demo")

client_id = st.text_input("Enter your search client_id:", placeholder="002")
if client_id:
    st.write(f"Showing recommendations for: **{client_id}**")

    results = get_recommendations(client_id)

    if len(results) > 0:
        st.dataframe(results)
    else:
        st.info("No matching results found.")