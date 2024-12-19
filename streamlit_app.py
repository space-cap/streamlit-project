import streamlit as st
from openai import OpenAI

# Show title and description.
st.title("💬 Assignment 6")

st.write(
    "Migrate the RAG pipeline you implemented in the previous assignments to Streamlit. \n"
    "Implement file upload and chat history. \n"
    "Allow the user to use its own OpenAI API Key, load it from an st.input inside of st.sidebar \n"
    "Using st.sidebar put a link to the Github repo with the code of your Streamlit app. \n"
)