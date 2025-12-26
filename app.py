import os
from dotenv import load_dotenv

from flask import Flask, jsonify, render_template, request
from src.helper import download_hf_embeddings
from src.prompt import system_prompt

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = Flask(__name__)

_ = load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
LLM_API = os.getenv('LLM_API')

BASE_URL = "https://openrouter.ai/api/v1"

MODEL = 'meta-llama/llama-3.3-70b-instruct:free'

embeddings = download_hf_embeddings()

index_name = "medical-bot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever =docsearch.as_retriever(search_type='similarity', search_kwargs={'k': 3})

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=LLM_API,
    model=MODEL,
    temperature=0.4,
    max_tokens=500
)

prompt = ChatPromptTemplate.from_messages(
    [
        ('system', system_prompt),
        ('human', '{input}')
    ]
)

rag_chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt
    | llm                                                   
    | StrOutputParser()
)

@app.route('/')
def index():
    render_template('chat.html')

@app.route('/get', methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    input = msg
    print(input)
    response = rag_chain.invoke(msg)
    print(response)
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

