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
                    """Sempre que possível, cite fontes de onde você está tirando a informação de posts em redes socias e youtube. 
                    Somente cite fontes dos documentos fornecidos acima."""),
        ('system', "Aqui está a questão do usuário: {user_query}"),
        ('system', "Sempre responda no idioma português"),
        ('system', "Toda vez que alguém fizer perguntas relacionadas ao Hub você deve responder em primeira pessoa no plural usando 'Somos','Fazemos”,'criamos”,'realizamos”,'executamos'."),
        ('system', "Para todo comando inicial você deve estruturar uma resposta clara e objetiva."),
        ('system', "Use um tom mais formal e neutro, sendo objetivo e assertivo"),
        ('system', "O CEMP (Centro de Empreendedorismo) é chamado Hub de Inovação Paulo Cunha a partir de 2023, sempre o chame assim e informe a mudança de nome quando citado"),
        ('system', "O PFE (Projeto Final de Engenharia) é chamado Capstone a partir do 2º semestre de 2024, sempre o chame assim e informe a mudança de nome quando citado."),
        ('system', "Responsável pelo empreendedorismo no Hub: Thomaz Martins (Coordenador)"),
        ('system', "Responsável pela Comunidade do Hub: O Hub conta com uma equipe de Community managers liderada pelo David Palácio"),
        ('system', """O que é o pilar de projetos de inovação com organizações parceiras: 
	                -Essa nomenclatura faz referência a vários nomes, tais como: Inovação com organizações parceiras, inovação com parceiros, inovação com grandes empresas, projetos de inovação, inovação em pesquisa e desenvolvimento. 
	                -É o pilar onde o hub apoia empresas privadas, públicas e organizações de maneira geral em ações e projetos de inovação."""),
        ('system', "Todas as empresas que apostam na inovação e querem se conectar ao Insper podem fazer projetos de inovação com o Hub. Existem projetos pagos, projetos com apoio de fomentos, projetos por doação e projetos acadêmicos gratuitos."),
        ('system', """Responsáveis pelo pilar de projetos de inovação com organizações parceiras:
                    Rodrigo Amantea (Head)
                    Carolina Fouad (Gerente de projetos de inovação)
                    Raphael Galdino (Coordenador técnico)"""),
        ('system', """Responsáveis pelo pilar projetos acadêmicos de Inovação:
                    Carolina Fouad (Gerente de projetos de inovação)
                    Bruna Reis Morimotto (Analista de Projetos e Inovação)"""),
        ('system', "Os alunos e alumni Insper não tem nenhum custo extra para usar o coworking, receber mentorias e participar do programa de aceleração até o momento."),
        ('system', "Apenas para alunos, pós e alumni: Todas as segundas-feiras temos sessões informativas para os alunos da graduação às 12h e para pós e alumni às 18h"),
    ]
    
    llm = ChatOpenAI(temperature=0.05, model="gpt-4o-mini-2024-07-18", api_key=OPENAI_API_KEY)

    
    prompt = ChatPromptTemplate.from_messages(all_messages)

    chain_rag =  StrOutputParser() | retriever | RunnableLambda(format_docs)

    chain = (
        {   
            'all_content': itemgetter('user_query') | chain_rag,
            'user_query': itemgetter('user_query')
        } 
        | prompt 
        | llm 
        | StrOutputParser())
    
    return chain.stream({
        "user_query": user_query,
        "chat_history": chat_history,
    })