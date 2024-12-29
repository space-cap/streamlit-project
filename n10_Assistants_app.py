import streamlit as st
import google.generativeai as genai
from langchain.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
import yfinance
import json
import os


st.set_page_config(
    page_title="Assistants",
    page_icon="🖥️",
)

st.markdown(
    """
    # Assistants
            
    genai 로 비슷하게 하려고 했지만 genai에서는 할 수 없었습니다.

"""
)

def get_ticker(inputs):
    ddg = DuckDuckGoSearchAPIWrapper()
    company_name = inputs["company_name"]
    return ddg.run(f"Ticker symbol of {company_name}")

def get_income_statement(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.income_stmt.to_json())

def get_balance_sheet(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.balance_sheet.to_json())

def get_daily_stock_performance(inputs):
    ticker = inputs["ticker"]
    stock = yfinance.Ticker(ticker)
    return json.dumps(stock.history(period="3mo").to_json())

# Define the functions as a list of dictionaries
functions = [
    {
        "name": "get_ticker",
        "description": "Given the name of a company returns its ticker symbol",
        "parameters": {
            "type": "object",
            "properties": {
                "company_name": {
                    "type": "string",
                    "description": "The name of the company",
                }
            },
            "required": ["company_name"],
        },
    },
    {
        "name": "get_income_statement",
        "description": "Given a ticker symbol (i.e AAPL) returns the company's income statement.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker symbol of the company",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_balance_sheet",
        "description": "Given a ticker symbol (i.e AAPL) returns the company's balance sheet.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker symbol of the company",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_daily_stock_performance",
        "description": "Given a ticker symbol (i.e AAPL) returns the performance of the stock for the last 100 days.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker symbol of the company",
                },
            },
            "required": ["ticker"],
        },
    },
]

with st.sidebar:
  
    st.markdown("""
    소스 코드
                
    https://github.com/space-cap/streamlit-project/blob/main/n10_Assistants_app.py
    """)
  
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

if google_api_key:
    genai.configure(api_key=google_api_key)

    model = genai.GenerativeModel('gemini-1.5-flash')

    def generate_response(user_input):
        try:
            # 지시사항을 사용자 입력의 일부로 포함
            full_prompt = f"""지시사항: 당신은 투자 조언자입니다. 공개 거래되는 회사에 대한 연구를 도와주고 사용자가 주식을 살지 말지 결정하는 데 도움을 줍니다. 다음 함수들을 사용할 수 있습니다:

    1. get_ticker: 주어진 회사 이름의 티커 심볼을 찾습니다.
    2. get_income_statement: 티커 심볼을 사용하여 회사의 손익계산서를 가져옵니다.
    3. get_balance_sheet: 티커 심볼을 사용하여 회사의 대차대조표를 가져옵니다.
    4. get_daily_stock_performance: 티커 심볼을 사용하여 지난 100일간의 주식 성과를 가져옵니다.

    이 함수들을 사용하여 회사와 그들의 주식에 대한 정확하고 도움이 되는 정보를 제공하세요.

    사용자 질문: {user_input}"""

            response = model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            return f"오류가 발생했습니다: {str(e)}"

    user_query = "What's the current financial state of Apple?"
    response = generate_response(user_query)
    st.write(response)




