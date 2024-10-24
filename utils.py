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

# def format_docs(docs):
#     return "\n\n".join([d.page_content for d in docs])

def format_docs(docs):
    out_doc = "\n\n"
    for d in docs:
        out_doc += d.page_content + "\n"
        for index in d.metadata:
            out_doc += index + ": " + str(d.metadata[index]) + "\n"
        out_doc += "\n\n"
    return out_doc

# def run_rag(user_query):
#     if user_query != str:
#         user_query = user_query.messages[0].content
#     context = db.similarity_search(user_query, k=3)
#     all_content = "\n\n".join(get_strings_from_documents(context))
#     return all_content

def cs_sidebar():
    st.sidebar.header("HUBot - Sua IA para inovadores e empreendedores")

    st.sidebar.markdown('https://www.hub.insper.edu.br/')

    st.sidebar.markdown('https://www.insper.edu.br/')

    st.sidebar.markdown("""
                        <div style='display: flex; flex-direction: column; justify-content: space-between; height: 100%;'>
                            <div style='margin-top: 10rem;'>
                                <h4> About <a href='https://neroai.com.br/'> 
                                                <img src='data:image/png;base64,{}' class='img-fluid' height=70>
                                            </a> </h4>
                                <h4> Powered by <a href='http://hub.insper.edu.br/'>
                                    <img src='data:image/png;base64,{}' class='img-fluid' width=112 height=20>
                                </a> </h4>
                            </div>
                        </div>
                        """.format(img_to_bytes("imgs/logo_hub.png"), img_to_bytes("imgs/neroai_logo.png")), unsafe_allow_html=True)

    return None
def get_retriever(statement, embeddings):
    vdb_path = f"vectorstore/hub_{statement}"
    db = FAISS.load_local(vdb_path, embeddings, allow_dangerous_deserialization=True)
    return db

def classfifier_rag(query):
    rules_verifier = """
    Você é um agente classificador de querys. Seu trabalho é classificar a intenção do usuário através de sua pergunta.

    As classificações possíveis são: "institucional", "empreendedorismo", "projects", "webinars" e "news_social".

    Responda apenas uma palavra com a classificação da query.

    Para fazer a classificação, você deve seguir as seguintes regras:

    1. Fazem parte da classificação "institucional" perguntas que pedem informações sobre:
        De forma direta:
        - Documentos institucionais do Hub de Inovação, como regimentos, manuais, portarias, resoluções, entre outros.
        - Informações sobre o Hub de Inovação, como história, missão, visão, valores, entre outros.
        - Pessoas que fazem parte do Hub de Inovação, como coordenadores, gerentes, analistas, entre outros.
        De forma mais geral:
        - Projetos e programas do Hub de Inovação, como projetos de inovação, projetos acadêmicos, programas de aceleração, entre outros.
        - Eventos do Hub de Inovação, como webinars, palestras, workshops, entre outros.

    2. Fazem parte da classificação "empreendedorismo" perguntas que pedem informações sobre:
        - Empreendedorismo em geral, como definição, conceitos, teorias, entre outros.
        - Empreendedorismo no contexto do Hub de Inovação, como programas, eventos, projetos, entre outros.
        - Atuação do Hub de Inovação no empreendedorismo, como apoio a startups, mentoria, aceleração, entre outros.

    3. Fazem parte da classificação "projects" perguntas que pedem informações gerais e específicas sobre Oportunidades de Negócios "ON", Resolução Eficaz de Problemas "REP", Women in Tech "WIT" , Projeto Final de Engenharia "PFE" ou Capstone.
        - O REP é uma oportunidade do Insper para o curso de Administração, onde os alunos podem resolver problemas reais de empresas no lugar de produzir um TCC.
        - O Women in Tech é uma iniciativa do Hub de Inovação que busca incentivar a participação feminina na área de tecnologia.
        - O PFE é um projeto de conclusão de curso para os alunos de Engenharia do Insper, agora renomeado Capstone.
    
    4. Para perguntas que pedem informações diretas sobre webinars e vídeos do Youtube, você deve retornar a mensagem "webinars".
    5. Para perguntas que pedem informações diretas sobre notícias ou posts em redes sociais, você deve retornar a mensagem "news_social".


    Se não se encaixar em nenhuma classificação, você deve retornar a mensagem "institucional".

    A query feita pelo usuário foi:
    {query}

    A classificação da query é:
    """

    llm = ChatOpenAI(temperature=0.02, model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_messages([('system', rules_verifier)])

    ver_agent = prompt | llm | StrOutputParser()

    statement = ver_agent.invoke(input={"query": query})
    return statement

def respond(user_query, chat_history, retriever, statement, retriever_context=None):
    
    
    if statement == "institucional":
        rag_rule = "Aqui está o contexto adicional de documentos institucionais: {all_content}" +  "\n\n" + \
                """Sempre que possível, cite fontes de onde você está tirando a informação de posts em redes socias e youtube. 
                Somente cite fontes dos documentos fornecidos acima."""
    else:
        rag_rule = "Aqui está o contexto adicional de documentos institucionais: {all_content}" +  "\n\n" + "{rag_context}" + \
                """Sempre que possível, cite fontes de onde você está tirando a informação de posts em redes socias e youtube. 
                Somente cite fontes dos documentos fornecidos acima."""
    all_messages = [
        ('system', "Aqui está o que foi conversado até agora:\n\n" + \
                    "\n\n".join([msg.content for msg in chat_history[-4:]])),
        ('system', """
                    Você é um assistente do Hub de inovação do Insper. 
                    Você vai responder perguntas sobre Startups e Empreendedorismo. 
                    Se apresente e diga como você pode ajudar."""),
        ('system', rag_rule),
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
        ('system', "Caso a questão do usuário seja relacionado à empreendedorismo, ou ele tenha interesse em empreender, cadastrar sua empresa ou solicitar ajuda com algum projeto no Hub, indique a incrição através do formulário para contato com a equipe de Advisory com o link: https://forms.gle/3QPbUKsrfsVVazf48"),
        ('system', "Caso o usuário pergunte pelos contatos do Hub, informe: Email: hub@insper.edu.br, WhatsApp: https://wa.me/message/SNMDWEXHGB7AN1, Website: http://www.insper.edu.br/hub"),
        ('system', "Se o usuário perguntar sobre eventos futuros do Hub, apenas para ele entrar em contato com o Hub para mais informações."),
    ]
    
    llm = ChatOpenAI(temperature=0.05, model="gpt-4o-mini-2024-07-18", api_key=OPENAI_API_KEY)

    
    prompt = ChatPromptTemplate.from_messages(all_messages)

    chain_rag =  retriever | format_docs

    if statement == "institucional":
        chain = (
            {   
                'all_content': itemgetter('user_query') | chain_rag,
                'user_query': itemgetter('user_query')
            } 
            | prompt 
            | llm 
            | StrOutputParser())
    else:
        chain_rag_context =  retriever_context |format_docs
        chain = (
        {   
            'all_content': itemgetter('user_query') | chain_rag,
            'rag_context': itemgetter('user_query') | chain_rag_context,
            'user_query': itemgetter('user_query')
        } 
        | prompt 
        | llm 
        | StrOutputParser())

    
    return chain.stream({
        "user_query": user_query,
        "chat_history": chat_history,
    })

from unidecode import unidecode

def formata(s):
    # Remove acentos
    s = unidecode(s)
    # Transforma tudo em minúsculas
    s = s.lower()
    return s