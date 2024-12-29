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
                
    https://github.com/space-cap/streamlit-project/blob/main/n04_SiteGPT_app.py
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


genai.configure(api_key=google_api_key)

model = genai.GenerativeModel('gemini-1.5-flash')

instructions = """You are an Investor Assistant. You help users do research on publicly traded companies and you help users decide if they should buy the stock or not. You have access to the following functions:

1. get_ticker: Find the ticker symbol for a given company name.
2. get_income_statement: Get the income statement for a company using its ticker symbol.
3. get_balance_sheet: Get the balance sheet for a company using its ticker symbol.
4. get_daily_stock_performance: Get the stock performance for the last 100 days using a ticker symbol.

Use these functions to provide accurate and helpful information to users about companies and their stocks."""

# Function to generate a response
def generate_response(user_input):
    response = model.generate_content([
        {"role": "system", "parts": [instructions]},
        {"role": "user", "parts": [user_input]}
    ])
    return response.text

# Example usage
user_query = "Research about the XZ backdoor."
response = generate_response(user_query)
st.write(response)




