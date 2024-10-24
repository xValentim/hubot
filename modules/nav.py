import streamlit as st
from streamlit import session_state as ss


def HomeNav():
    st.sidebar.page_link("app.py", label="Home", icon='🏠')


def LoginNav():
    st.sidebar.page_link("pages/account.py", label="Account", icon='🔐')


def HubotNav():
    st.sidebar.page_link("pages/chatbot.py", label="Hubot", icon='🤖')


def MenuButtons():
    if 'authentication_status' not in ss:
        ss.authentication_status = False

    # Sempre mostra a HOME e LOGIN.
    HomeNav()
    LoginNav()

    # Se o usuário logar, mostra as demais telas.
    if ss["authentication_status"]:
        HubotNav()
