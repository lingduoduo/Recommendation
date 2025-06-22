import streamlit as st

st.page_link("app.py", label="Home", icon="🏠")

name = st.text_input("Name - Text Input")

choice = st.number_input("Choice - Number Input")

number = st.slider("Pick a number - Slider Input", 0, 100)

text = st.text_area("Text - Text Area")

size = st.select_slider("Pick a size", ["S", "M", "L"])

if st.button('Say hello'):
    st.write('Why hello there')
else:
    st.write('Goodbye')

date = st.date_input("Date - Date Input")

time = st.time_input("Time - Time Input")


if st.button("🔁 Rerun this page"):
    st.rerun()
