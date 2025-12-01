import streamlit as st

if st.session_state.key:
    st.write(f"Your key: {st.session_state.key}")

container = st.container()
if 'message' not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input()
if prompt:
    st.session_state.messages.append(prompt)

with container:
    with st.chat_message("user"):
        for message in st.session_state.messages:
            st.write(message)
