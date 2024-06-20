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

load_dotenv()

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

embedding_size = 3072
embedding_model = 'text-embedding-3-large'
embeddings = OpenAIEmbeddings(model=embedding_model)

# app config
st.set_page_config(page_title="Streamlit Chatbot", page_icon="🤖")
st.title("Assistente virtual - Hub Insper")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Olá, eu sou o assistente do Hub de inovação do Insper. Estou aqui para responder perguntas sobre Startups e Empreendedorismo. Como posso ajudar você?"),
    ]

if 'db' not in st.session_state:
    st.session_state.db = FAISS.load_local("vectorstore/blend_yc_faiss_index", embeddings, allow_dangerous_deserialization=True)
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

    st.session_state.chat_history.append(AIMessage(content=response))