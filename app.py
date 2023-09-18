import streamlit as st
import os
import openai
import requests
import re
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders.base import Document
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import validators
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate


openai.api_key = os.environ.get("OPENAI_API_KEY")

# Build prompt
template = """Answer to the user queries like a chatbot agent. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)


def docPrePro(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 500,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    documents = text_splitter.split_documents(doc)
    db = FAISS.from_documents(documents, OpenAIEmbeddings())

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-0301", temperature=0)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        memory = memory,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    return qa_chain




# Function to scrape only visible text from the given URL
def scrape_visible_text_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove script, style, and other non-visible tags
    for tag in soup(["script", "style", "meta", "link", "noscript", "header", "footer", "aside", "nav", "img"]):
        tag.extract()

    # Get the header content
    header_content = soup.find("header")
    header_text = header_content.get_text() if header_content else ""

    # Get the paragraph content
    paragraph_content = soup.find_all("p")
    paragraph_text = " ".join([p.get_text() for p in paragraph_content])

    # Combine header and paragraph text
    visible_text = f"{header_text}\n\n{paragraph_text}"

    # Remove multiple whitespaces and newlines
    visible_text = re.sub(r'\s+', ' ', visible_text)
    return visible_text.strip()


qa = RetrievalQA

st.title("Langchain - App")
url = st.sidebar.text_input('URL Of data')
pdf = st.sidebar.file_uploader("Upload a file", type=["pdf"])
if url:
    if validators.url(url):
        text = scrape_visible_text_from_url(url)
        d = []
        d.append(Document(page_content=text))
        qa = docPrePro(d)
    else:
        st.warning("Enter a valid url")
elif pdf:
    destination = os.path.join('uploads/',pdf.name)
    with open(os.path.join("uploads/",pdf.name),"wb") as f:
        f.write(pdf.getbuffer())
    loader = PyPDFLoader(destination)
    d = loader.load()
    qa = docPrePro(d)
    os.remove(destination)
else:
     st.warning("Please upload a file or submit a url before proceeding.")






if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    result = qa({"query": prompt})
    # response = f"Echo: {result['result']}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(result['result'])
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": result['result']})