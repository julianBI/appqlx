import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

import os
from pathlib import Path

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.session_state.conversation = get_conversation_chain(vectorstore)

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    st.session_state.chat_history.reverse()  # Reverse the order of chat history

    # Display user question
    #st.markdown(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)

    # Display bot answers
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    #load_dotenv()
        # Set page config
    st.set_page_config(
        page_title="Qualex Consulting Services - AI -",
        page_icon="🤖"
    )

    # Custom HTML for the title with an image
    image_url = "https://media.licdn.com/dms/image/C4E0BAQE2UVRGwPof-g/company-logo_200_200/0/1674677654645/qualexconsulting_logo?e=2147483647&v=beta&t=g1dAO5Y3vrl1BDfkYDKj2-vqB96DRnnu8ND3qy0ck8Y"
    title_text = "Qualex Consulting Services - AI -"

    st.markdown(f'<div style="display: flex; align-items: center;">'
                f'<img src="{image_url}" alt="Qualex Logo" style="width: 50px; margin-right: 10px;">'
                f'<h1>{title_text}</h1>'
                '</div>', unsafe_allow_html=True)

    st.write(css, unsafe_allow_html=True)

     # Add a password input in the sidebar for OPENAI_API_KEY
    OPENAI_API_KEY = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

    st.header("Chat with multiple PDFs :books:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
        # Set the OPENAI_API_KEY in the environment variable
        # Set the OPENAI_API_KEY in the environment variable
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if __name__ == '__main__':
    main()
