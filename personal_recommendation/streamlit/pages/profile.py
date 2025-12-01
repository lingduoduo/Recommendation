import streamlit as st
import pandas as pd
import numpy as np

st.write("This is a profile page.")

chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns = ['a', 'b', 'c'])

st.write(chart_data)

import time
def my_generator():
    for i in range(10):
        yield f'{i}'
        time.sleep(.5)
st.write_stream(my_generator)

# magic_data
# chart_data

st.markdown("Test Streamlit App")

st.title("Profile Page")

st.header("This is a header")

st.code("a = 12345")
st.code("print(a)")
st.divider()

st.json({'key1': 'value1', "key2": "value2"})

if st.button("Say Hello"):
    st.write("Hello World")
else:
    st.write("Goodbye")

text_content = st.text_area("Text")
st.download_button("Download some data", text_content)

st.link_button("Google", "https://google.com")

with st.sidebar:
    st.page_link("app.py", label="Home Page", icon="🏠")
    st.page_link("pages/profile.py", label="👤 My Profile")
    # st.page_link("pages/auto_complete.py", label="🔤 Auto Complete")
    # st.page_link("pages/cold_start.py", label="❄️ Cold Start")
    # st.page_link("pages/content_recommendation.py", label="📚 Content Recommendation")

selected = st.checkbox("Checkbox")
st.write(f"You selected: {selected}")

activated = st.toggle("Activate")
st.write(f"You activated: {activated}")

radio = st.radio("Pick one", ["chooce1", "chooce2"])
st.write(f"You selected: {radio}")

choice = st.selectbox("Pick one", ["chooce1", "chooce2"])
st.write(f"You selected: {choice}")

choices = st.multiselect("Select one", ["chooce1", "chooce2", "chooce3"])
st.write(f"You selected: {choices}")

number = st.slider("Pick a number", 0, 10)
st.write(f"You selected: {number}")

size = st.select_slider("Pick a size", options=[1, 2, 3, 4, 5, 6, 7])
st.write(f"You selected: {size}")



