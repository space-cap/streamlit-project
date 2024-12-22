from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser
from langchain.storage import InMemoryStore
import streamlit as st
import json
import os
import tempfile


st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("💬 Assignment 7")

st.markdown("""
QuizGPT를 구현하되 다음 기능을 추가합니다:
            
함수 호출을 사용합니다.
            
1. 유저가 시험의 난이도를 커스터마이징 할 수 있도록 하고 LLM이 어려운 문제 또는 쉬운 문제를 생성하도록 합니다.
2. 만점이 아닌 경우 유저가 시험을 다시 치를 수 있도록 허용합니다.
3. 만점이면 st.ballons를 사용합니다.
4. 유저가 자체 OpenAI API 키를 사용하도록 허용하고, st.sidebar 내부의 st.input에서 로드합니다.
5. st.sidebar를 사용하여 Streamlit app의 코드와 함께 Github 리포지토리에 링크를 넣습니다.
""")




template = """
다음 텍스트를 기반으로 4지선다형 문제 10개를 만들어주세요:

{text}

문제와 답변을 다음 JSON 형식으로 제공해주세요:
{{ "questions": [
            {{
                "question": "What is the color of the ocean?",
                "answers": [
                        {{
                            "answer": "Red",
                            "correct": false
                        }},
                        {{
                            "answer": "Yellow",
                            "correct": false
                        }},
                        {{
                            "answer": "Green",
                            "correct": false
                        }},
                        {{
                            "answer": "Blue",
                            "correct": true
                        }},
                ]
            }},
        ...
    ]
}}
"""

prompt = ChatPromptTemplate.from_template(template)


@st.cache_resource(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    loader = UnstructuredFileLoader(file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs


@st.cache_resource(show_spinner="Loading file...")
def split_file_cloud(file):
    with tempfile.TemporaryDirectory() as temp_dir:
        # 임시 파일 경로 생성
        temp_file_path = os.path.join(temp_dir, file.name)
        
        # 임시 파일에 내용 쓰기
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.getbuffer())
        
        st.write(f"임시 파일 경로: {temp_file_path}")

        # 메모리 내 저장소 사용
        cache_dir = InMemoryStore()
        
        loader = UnstructuredFileLoader(temp_file_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        return docs



@st.cache_resource(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


@st.cache_resource(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    # 전체 텍스트 결합
    full_text = " ".join([doc.page_content for doc in docs])
    
    # LLM을 사용하여 문제 생성
    response = llm.invoke(prompt.format(text=full_text))
    cleaned_string = response.content.replace('```', '').replace('json', '', 1).strip()
    
    # JSON 파싱
    questions_json = json.loads(cleaned_string)
    return questions_json


with st.sidebar:
    st.markdown("""
    구글 키 가지고 오기
    https://aistudio.google.com/apikey
    """)
    
    # API Key 입력
    google_api_key = st.text_input("Input your Google API Key", type="password")
    if not google_api_key:
        st.info("Please add your Google API key to continue.", icon="🗝️")
        st.stop()
    else:
        st.write("key ok")

    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file_cloud(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


# LLM 초기화
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.1,
    api_key=google_api_key,
)

