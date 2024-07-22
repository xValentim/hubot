from dotenv import load_dotenv
import streamlit as st
import os

# Langchain
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from operator import itemgetter
import base64
from pathlib import Path

from utils import *

load_dotenv()


def final_func():
    response1 = "Se precisar de mais informações ou quiser falar diretamente com nossa equipe, entre em contato:\n- **Email:** hubinovacao@insper.edu.br\n- **Telefone:** (11) 1234-5678\n- **Website:** [www.insper.edu.br/hub](http://www.insper.edu.br/hub) \n\nObrigado por utilizar o chatbot do Hub de Inovação do Insper!"   
    st.session_state.chat_history.append(AIMessage(content=response1))

embedding_size = 3072
embedding_model = 'text-embedding-3-large'
embeddings = OpenAIEmbeddings(model=embedding_model)

# app config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Assistente virtual - Hub Insper")
cs_sidebar()

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, eu sou o assistente do Hub de inovação do Insper. Estou aqui para responder perguntas sobre Startups e Empreendedorismo. Como posso ajudar você?"),
    ]

if 'db' not in st.session_state:
    st.session_state.db = FAISS.load_local("vectorstore/hub_index", embeddings, allow_dangerous_deserialization=True)
    st.session_state.retriever = st.session_state.db.as_retriever()

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="🤖"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="👤"):
            st.write(message.content)

# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human", avatar="👤"):
        st.markdown(user_query)

    with st.chat_message("AI", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = st.write_stream(respond(user_query, st.session_state.chat_history, st.session_state.db, st.session_state.retriever))
            st.write("Mais alguma dúvida?")        
            st.button('Encerrar Chat', on_click=final_func)
    st.session_state.chat_history.append(AIMessage(content=response))