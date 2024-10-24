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
# Tela de login
from streamlit import session_state as ss
from modules.nav import MenuButtons

if 'authentication_status' not in ss:
    st.switch_page('./pages/account.py')

MenuButtons()


load_dotenv()

st.markdown(
    """
    # Hubot
    """,
    unsafe_allow_html=True
)

@st.dialog("Seja bem-vindo(a) ao chatbot do Hub de Inovação do Insper!")
def vote():
    response1 = "Ao final dessa conversa, se precisar de mais informações ou quiser falar diretamente com nossa equipe, entre em contato:\n- **Email:** hub@insper.edu.br\n- **Telefone:** +55 11 98251-0087\n- **Website:** [hub.insper.edu.br/](http://hub.insper.edu.br) \n\n Recarregue a página ou peça novamente a informação ao Hubot para rever os contatos.  "   
    st.write(response1)

embedding_size = 3072
embedding_model = 'text-embedding-3-large'
embeddings = OpenAIEmbeddings(model=embedding_model)

# app config
st.title("Inteligência Artificial do Hub - HUBot")
cs_sidebar()

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, sou HUBot, a IA do Hub de inovação e Empreendedorismo Paulo Cunha do Insper. Estou aqui para responder perguntas sobre o ecossistema de inovação, empreendedorismo e startups do Insper. Como posso te ajudar?"),
    ]

if 'db' not in st.session_state:
    st.session_state.aux = False
    st.session_state.db = FAISS.load_local("vectorstore/hub_institucional", embeddings, allow_dangerous_deserialization=True)
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

user_query = st.chat_input("Digite algo...")

if not st.session_state.aux:
    vote()
    st.session_state.aux = True
    
if user_query is not None and user_query != "":
    
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human", avatar="👤"):
        st.markdown(user_query)

    
    with st.chat_message("AI", avatar="🤖"):
        with st.spinner("Thinking..."):
            statement = classfifier_rag(user_query)
            if statement != "institucional":
                st.session_state.db_context = get_retriever(statement, embeddings)
                st.session_state.retriever_context = st.session_state.db_context.as_retriever()
                response = st.write_stream(respond(user_query, st.session_state.chat_history, st.session_state.retriever, statement, st.session_state.retriever_context))
            else:
                response = st.write_stream(respond(user_query, st.session_state.chat_history, st.session_state.retriever, statement))
            st.session_state.aux = True
            response2 = "Você tem mais alguma dúvida?"
            st.write(response2)
            st.session_state.chat_history.append(AIMessage(content=response))
            st.session_state.chat_history.append(AIMessage(content=response2))
            
    

