import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Assignment 6")

st.markdown("""
1. Migrate the RAG pipeline you implemented in the previous assignments to Streamlit.
2. Implement file upload and chat history.
3. Allow the user to use its own OpenAI API Key:
   - Load it from an `st.input` inside of `st.sidebar`.
4. Using `st.sidebar`, put a link to the Github repo with the code of your Streamlit app.
---
""")

st.markdown("""
구글 키 가지고 오기
https://aistudio.google.com/apikey
""")

google_api_key = st.text_input("Google API Key", type="password")
if not google_api_key:
    st.info("Please add your Google API key to continue.", icon="🗝️")
else:
    st.write("key ok")

