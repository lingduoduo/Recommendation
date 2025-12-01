import streamlit as st
import numpy as np

st.page_link("app.py", label="Home", icon="🏠")
st.page_link("pages/profile.py", label="My profile")

if "key" not in st.session_state:
    st.session_state.key = ""
key = st.text_input("Enter your key:", value=st.session_state.key)
st.session_state.key = key
st.success(f"Saved key: {st.session_state.key}")

st.page_link("pages/session_state.py", label="Sessions")

st.set_page_config(
    page_title = "Streamlit App",
    page_icon=":shark:"
)
st.query_params['key'] ='value'
st.query_params.clear()

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

data = st.file_uploader("Upload a csv")

st.text("Image")
st.image("https://images.unsplash.com/photo-1506744038136-46273834b3fb?ixlib=rb-4.0.3&auto=format&fit=crop&w=800&q=60",
         caption="A scenic mountain landscape", use_column_width=True)

st.text("Audio")
st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3")

st.text("video")
st.video("https://www.w3schools.com/html/mov_bbb.mp4")

st.sidebar.write("This lives in the sidebar")
st.sidebar.button("click me")

st.text("col1 and col2")
col1, col2 = st.columns(2)
col1.write("this is column1")
col1.button("click bottom1")
col2.write("this is column2")
col2.button("click bottom2")

tab1, tab2 = st.tabs(["Tab1", "Tab2"])
tab1.write("Tab - This is the first tab")
tab2.write("Tab - This is the second tab")

with st.expander("Open to see more"):
    st.write("Expander - This is more content")
    st.write("Expander - This is another content")

c = st.container()
st.write("Container - This will show last")
c.write("Container - This will show first")
c.write("This will show second")

prompt = st.chat_input("Chat - This will show content ")
if prompt:
    st.write(prompt)

with st.chat_message("user"):
    st.write("Hello")
    st.line_chart(np.random.randn(100, 3))

with st.chat_message("ai"):
    st.write("I am an AI assistant")

st.write("This is status bar")
import time
def do_something():
    time.sleep(1)
bar = st.progress(0)
for i in range(10):
    progress = (i + 1) * 10
    bar.progress(progress)
    do_something()
st.balloons()


with st.spinner("Please wait..."):
    time.sleep(1)
st.snow()

e = RuntimeError("This is an exception of type RuntimeError")
st.exception(e)
st.error("This is an exception of type RuntimeError")
st.warning("This is an warning of type RuntimeError")
st.info("This is an information of type RuntimeError")
st.success("This is an success message")


with st.status("🍳 Cooking..."):
    time.sleep(1)
    st.toast("🧈 Butter!", icon="🧈")

if st.session_state.get("clicked", False):
    st.write("You can click to rerun the app")
    if st.button("🔁 Rerun this page"):
        st.rerun()
    st.session_state["clicked"] = True
else:
    st.write('Goodbye')

st.stop()
