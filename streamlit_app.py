import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import InMemoryStore
import tempfile
import os

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


# LLM 초기화
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.1,
    api_key=google_api_key,
)


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


@st.cache_resource(show_spinner="Embedding file...")
def embed_file_from_cloud(file):
    with tempfile.TemporaryDirectory() as temp_dir:
        # 임시 파일 경로 생성
        temp_file_path = os.path.join(temp_dir, file.name)
        
        # 임시 파일에 내용 쓰기
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.getbuffer())
        
        st.write(f"임시 파일 경로: {temp_file_path}")

        # 메모리 내 저장소 사용
        cache_dir = InMemoryStore()

        splitter = CharacterTextSplitter.from_tiktoken_encoder(
            separator="\n",
            chunk_size=600,
            chunk_overlap=100,
        )

        loader = TextLoader(temp_file_path)
        docs = loader.load_and_split(text_splitter=splitter)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=google_api_key)
        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
        vectorstore = FAISS.from_documents(docs, cached_embeddings)
        retriever = vectorstore.as_retriever()
        return retriever




def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt file",
        type=["txt"],
    )


if file:
    retriever = embed_file_from_cloud(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    
else:
    st.session_state["messages"] = []



