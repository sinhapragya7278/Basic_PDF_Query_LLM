# Basic_PDF_Query_LLM_using_Langchain

# PDF Text Processing and Question Answering with LangChain and OpenAI

This repository contains a script for processing PDF files, extracting text, embedding the text using OpenAI embeddings, and performing question-answering using LangChain and OpenAI. The script is designed to be run in a Google Colab environment.

## Table of Contents
- [Installation](#installation)
- [Setup](#setup)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Installation

To install the necessary libraries, run the following commands:

```bash
pip install langchain
pip install openai
pip install PyPDF2
pip install faiss-cpu
pip install tiktoken

import os
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"

#load and read pdf

from PyPDF2 import PdfReader

# Location of the PDF file
reader = PdfReader('/content/gdrive/My Drive/SQL_basic.pdf')

# Extract text from the PDF
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# Display the first 100 characters of the extracted text
raw_text[:100]

#split text in chunks
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Display the number of chunks and the first two chunks
len(texts)
texts[0]
texts[1]

#download embeddings from open AI

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)
docsearch
#Perform Question Answering
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Example query
query = "what is Python"
docs = docsearch.similarity_search(query)
answer = chain.run(input_documents=docs, question=query)
print(answer)

#install dependencies

pip install langchain openai PyPDF2 faiss-cpu tiktoken
# This `README.md` file provides a comprehensive guide to setting up and using the script, including installation steps, setup instructions, and usage examples. Adjust the repository URL and any specific details as needed.


