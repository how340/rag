# Ollama serves as the backend to host the LLM
from langchain_community.llms import Ollama

# packages to help load in the pdf file. 
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from langchain.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup as Soup
from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX,
                                  SUFFIXES_TO_IGNORE_REGEX)

from config import *
import logging
import sys
import os

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

global conversation
conversation = None

def init_index():
    if not INIT_INDEX:
        logging.info("continue without initializing index")
        return

    # Load in local pdf.
    dir_path = "file_storage/"
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f[0] != '.']

    #iterate through the directory to load all pdf files.
    file_collection = []
    for i in dir_path:
        document = PyPDFLoader(
            file_path=dir_path + files[0]
        ).load()
        file_collection.append(document)

    # split text
    # this chunk_size and chunk_overlap effects to the prompt size
    # execeed promt size causes error `prompt size exceeds the context window size and cannot be processed`
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(documents)

    # create embeddings with huggingface embedding model `all-MiniLM-L6-v2`
    # then persist the vector index on vector db
    for file in file_collection:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=INDEX_PERSIST_DIRECTORY
        )

def init_conversation():
    global conversation

    # load index
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=INDEX_PERSIST_DIRECTORY,embedding_function=embeddings)

    # llama2 llm which runs with ollama
    # ollama expose an api for the llam in `localhost:11434`
    llm = Ollama(
        model="llama2",
        base_url="http://localhost:11434",
        verbose=True,
    )

    # create conversation
    conversation = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )


def chat(question, user_id):
    global conversation

    chat_history = []
    response = conversation({"question": question, "chat_history": chat_history})
    answer = response['answer']

    logging.info("got response from llm - %s", answer)

    # TODO save history

    return answer