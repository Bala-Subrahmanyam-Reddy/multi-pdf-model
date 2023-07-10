from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

load_dotenv()


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
    
    if not text_chunks:
        return None  
    
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore


def main():
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    st.title("Chat with multiple PDF'S 💬")

    pdf = st.file_uploader("Upload your PDF'S ", type='pdf', accept_multiple_files=True)

    if pdf:
            with st.spinner("Processing"):
                getText = get_pdf_text(pdf)
                getChunks = get_text_chunks(getText)
                if not getChunks:
                    st.write("No text chunks found. Please check your PDF or try another one.")
                    return
                knowledgeBase = get_vectorstore(getChunks)
                query = st.text_input('Ask a question ?')
                cancel_button = st.button('Cancel')
                if cancel_button:
                    st.stop()
                if query:
                    if knowledgeBase is None:
                        st.write("No knowledge base found. Please check your PDF or try another one.")
                        return
                    docs = knowledgeBase.similarity_search(query)
                    llm = OpenAI()
                    chain = load_qa_chain(llm, chain_type='stuff')
                    with get_openai_callback() as cost:
                        response = chain.run(input_documents=docs, question=query)
                        print(cost)
                    st.write(response)


if __name__ == "__main__":
    main()
