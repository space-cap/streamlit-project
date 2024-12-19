import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

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
    st.stop()
else:
    st.write("key ok")




@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever



# LLM 초기화
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.1,
    api_key=google_api_key,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

response = llm.invoke("대한민국의 수도는?")
with st.chat_message("ai"):
    st.markdown(response.content)



