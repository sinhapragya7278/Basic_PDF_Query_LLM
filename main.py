pip install langchain
pip install openai
pip install PyPDF2
pip install faiss-cpu
pip install tiktoken
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
# Get your API keys from openai, you will need to create an account.
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
import os
os.environ["OPENAI_API_KEY"] = ""
# connect your Google Drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
root_dir = "/content/gdrive/My Drive/"
# location of the pdf file/files.
reader = PdfReader('/content/gdrive/My Drive/SQL_basic.pdf')
# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
# raw_text
raw_text[:100]
# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits.

text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
len(texts)
texts[0]
texts[1]
# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)
docsearch
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
chain = load_qa_chain(OpenAI(), chain_type="stuff")
#output
#query = "what is Python"
#docs = docsearch.similarity_search(query)
#chain.run(input_documents=docs, question=query)
#Python is a high-level, interpreted programming language known for its simplicity and readability. It is versatile and flexible, with a strong community and library support. 
#It is widely used in various fields such as web development, data analysis, artificial intelligence, and scientific computing, and is especially prominent in data science and machine learning. 
#Its design philosophy emphasizes code readability and it is easy to learn and use, making it a popular choice for beginners and experienced programmers alike.
