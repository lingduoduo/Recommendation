#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : ling.huang@adp.com
@File    : auto_complete.py
"""
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import redis
import os
from typing import List, Tuple
import streamlit.components.v1 as components  # Fix unresolved reference


# --- Function to get suggestions ---
def get_suggestions(redis_client, query: str, max_results: int = 20, fuzzy: bool = True) -> List[Tuple[str, float]]:
    try:
        suggestions = redis_client.ft().sugget(
            'top_action_keywords',
            query,
            num=max_results,
            fuzzy=fuzzy,
            with_scores=True
        )
        if not suggestions:
            return []
        return [(s.string, s.score) for s in suggestions]
    except Exception as e:
        st.error(f"Error fetching suggestions: {e}")
        return []


# --- Set up Redis connection ---
_ = load_dotenv(find_dotenv())
REDIS_HOST = os.getenv("redis_host")
REDIS_PORT = os.getenv("redis_port")
REDIS_PASSWORD = os.getenv("redis_password")

try:
    r = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        password=REDIS_PASSWORD,
        decode_responses=True,
        ssl_cert_reqs="none",
        ssl=True
    )
    # Test connection
    if not r.ping():
        st.error("Failed to connect to Redis.")
except redis.ConnectionError as e:
    st.error(f"Redis connection error: {e}")

# --- Streamlit UI ---
st.title("Auto-Complete Demo")
suggestions = get_suggestions(r, "")
options_html = "".join(
    f"<option value='{item[0]} (score: {item[1]:.2f})' />"
    for item in suggestions
)

autocomplete_html = f"""
<html>
  <body>
    <input type="text" id="autocomplete" list="suglist" placeholder="Start typing..." style="width: 300px; padding: 6px; font-size: 16px;" oninput="sendValue(this.value)">
    <datalist id="suglist">
      {options_html}
    </datalist>

    <script>
      const sendValue = (val) => {{
        window.parent.postMessage({{ type: 'streamlit:componentValue', value: val }}, '*');
      }};
    </script>
  </body>
</html>
"""

# --- Declare return value from embedded component ---
selected = components.html(autocomplete_html, height=80)
print(selected)
