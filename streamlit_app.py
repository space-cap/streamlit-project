import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Assignment 6")

st.write(
    "Migrate the RAG pipeline you implemented in the previous assignments to Streamlit."
    "Implement file upload and chat history."
    "Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar"
    "Using st.sidebar put a link to the Github repo with the code of your Streamlit app."
)