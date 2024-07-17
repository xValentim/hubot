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

load_dotenv()

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Funções auxiliares
def get_strings_from_documents(documents):
    return [doc.page_content for doc in documents]

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# def run_rag(user_query):
#     if user_query != str:
#         user_query = user_query.messages[0].content
#     context = db.similarity_search(user_query, k=3)
#     all_content = "\n\n".join(get_strings_from_documents(context))
#     return all_content

def cs_sidebar():

    st.sidebar.markdown("""
                        # HUBOT
                        <div style='display: flex; flex-direction: column; justify-content: space-between; height: 100%;'>
                            <div style='margin-top: 20rem;'>
                                <h4> About <a href='https://www.02pelaeducacao.org/'> 
                                                <img src='data:image/png;base64,{}' class='img-fluid' width=112 height=20>
                                            </a> </h4>
                                <h4> Powered by <a href='https://neroai.com.br/'>
                                    <img src='data:image/png;base64,{}' class='img-fluid' height=70>
                                </a> </h4>
                            </div>
                        </div>
                        """.format(img_to_bytes("imgs/neroai_logo.png"), img_to_bytes("imgs/logo_hub.png")), unsafe_allow_html=True)

    return None

def respond(user_query, chat_history, db, retriever):
    
    
    
    all_messages = [
        ('system', "Aqui está o que foi conversado até agora:\n\n" + \
                    "\n\n".join([msg.content for msg in chat_history[-4:]])),
        ('system', """
                    Você é um assistente do Hub de inovação do Insper. 
                    Você vai responder perguntas sobre Startups e Empreendedorismo. 
                    Se apresente e diga como você pode ajudar."""),
        ('system', "Aqui está o contexto adicional de videos no YouYube: {all_content}" +  "\n\n" + \
                    """Sempre que possível, cite fontes (dados do YouTube) de onde você está tirando a informação. 
                    Somente cite fontes dos documentos fornecidos acima."""),
        ('system', "Aqui está a questão do usuário: {user_query_en}"),
    ]
    
    llm_0_temp = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo-0125", api_key=OPENAI_API_KEY)
    
    llm = ChatOpenAI(temperature=0.05, model="gpt-3.5-turbo-0125", api_key=OPENAI_API_KEY)
    
    prompt_en = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"Transcreva o que foi dito para o Inglês: {user_query}")
    ])
    
    prompt = ChatPromptTemplate.from_messages(all_messages)

    chain_en = prompt_en | llm_0_temp | StrOutputParser() | {"user_query_en": RunnablePassthrough()}
    
    # chain_rag = ChatPromptTemplate.from_template("{user_query_en}") | RunnableLambda(run_rag)
    chain_rag =  StrOutputParser() | retriever | RunnableLambda(format_docs)

    chain = (
        chain_en
        | {
            'all_content': itemgetter('user_query_en') | chain_rag,
            'user_query_en': itemgetter('user_query_en')
        } 
        | prompt 
        | llm 
        | StrOutputParser())
    
    return chain.stream({
        "user_query": user_query,
        "chat_history": chat_history,
    })