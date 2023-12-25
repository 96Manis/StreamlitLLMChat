import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS

from langchain.chains.question_answering import load_qa_chain

# Load environment variables
load_dotenv()

def process_text(text):
    # Split the text into chunks using Langchain's RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Convert the chunks of text into embeddings to form a knowledge base
    embeddings = OpenAIEmbeddings()
    vectorstores = FAISS.from_texts(chunks, embeddings)
    vectorstores.save_local("faiss_index")
    new_db = FAISS.load_local("faiss_index", embeddings)

    return new_db

# Streamlit web application title
st.title("Chat")

# File uploader for PDF documents
pdf = st.file_uploader('Upload your PDF Document', type='pdf')

# If a PDF is uploaded, process its text and create a knowledge base
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    vectorstores = process_text(text)

# Function to perform question-answering based on the provided query
def query_answer(query):
    docs = vectorstores.similarity_search(query)
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type='stuff')
    response = chain.run(input_documents=docs, question=query)
    return response

# Streamlit chat input for user to ask questions
prompt = st.text_input("Ask a question")
if prompt:
    st.write(prompt)
    result = query_answer(prompt)
    st.write(result)
