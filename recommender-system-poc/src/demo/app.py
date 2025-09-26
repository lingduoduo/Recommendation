#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : ling.huang@adp.com
@File    : app.py
"""
import streamlit as st


st.title("Welcome to Recommender System Demo Home Page 🏠")
st.write("Start from Use Cases below:")

st.info("1. Demo for Query Auto-Complete.")
st.page_link("pages/auto_complete.py", label="🔤 Auto-Complete")

st.info("2. Demo for Code Start Recommender.")
st.page_link("pages/cold_start.py", label="❄️ Cold Start")

st.info("3. Demo for Code Start Recommender.")
st.page_link("pages/content_based.py", label="📚 Content Based Recommendation")

st.balloons()
