from dotenv import load_dotenv
import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import DirectoryLoader
from langchain.chat_models import ChatOpenAI
import magic
import os
import nltk
import urllib.parse

def authenticate():
    with open('./config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )
    name, authentication_status, username = authenticator.login('Login', 'main')
    if st.session_state["authentication_status"]:
        authenticator.logout('Logout', 'main', key='unique_key')
        st.write(f'Welcome *{st.session_state["name"]}*')
        process()
    elif st.session_state["authentication_status"] is False:
        st.error('Username/password is incorrect')
    elif st.session_state["authentication_status"] is None:
        st.warning('Please enter your username and password')
    return name, authentication_status, username

@st.cache_data
def buildKnowledgeBase():
    load_dotenv()
    loader = DirectoryLoader('./docs/', glob='**/*.pdf')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_documents(texts, embeddings)
    print("Knowledge base built!")
    return knowledge_base

@st.cache_data
def retrieveKnowledgeBase():
    embeddings = OpenAIEmbeddings()
    knowledge = FAISS.load_local(".", embeddings, "mf-ug-index")
    return knowledge

def parse_response(response):
    response_sections = (response.split('SOURCES:'))
    response = response_sections[0]
    sources_section = response_sections[1]
    sources = sources_section.split(',')
    return response, sources

def hide_streamlit_menu_and_footer():
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            footer:after {
                content:'Made by Multifamily Architecture'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def process():
    doc_base = 'https://skarlekar.github.io/digital-sme-docs/'
    
    model_radio = st.sidebar.radio(
        "Choose a model to use:",
        ("gpt-3.5-turbo", "gpt-4")
    )

    #knowledge_base = buildKnowledgeBase()
    knowledge_base = retrieveKnowledgeBase()

    st.header("Ask me anything about Multifamily Underwriting 💬")
    user_question = st.text_input("Ask a question about Multifamily Underwriting:")
    if user_question:
        s = ''
        docs = knowledge_base.similarity_search(user_question)
        
        llm = ChatOpenAI(temperature=0, model=model_radio)
        chain = load_qa_with_sources_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
          
        parsed_response, sources = parse_response(response)   
        st.write(parsed_response) 
        st.markdown('**SOURCES:**')
        s = ''
        for i in sources:
            url = "{}{}".format(doc_base, urllib.parse.quote(i.strip()))
            s += "- [{}]({}) \n".format(i.strip(), url)
            #s += '- [' + i.strip() + '](' + doc_base + i.strip() + ')' + '\n'
            #s += "- " + i + "\n"
        st.markdown(s)

def main():
    st.set_page_config(page_title="Fannie Mae Multifamily Underwriting SME")
    hide_streamlit_menu_and_footer()
    name, authstatus, username = authenticate()
    

if __name__ == '__main__':
    main()