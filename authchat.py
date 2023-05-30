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
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
import magic
import os
import nltk

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

def process():

    knowledge_base = buildKnowledgeBase()

    st.header("Ask me anything about Multifamily Underwriting 💬")
    user_question = st.text_input("Ask a question about Multifamily Underwriting:")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)

def main():
    st.set_page_config(page_title="Fannie Mae Multifamily Underwriting SME")
    name, authstatus, username = authenticate()
    

if __name__ == '__main__':
    main()