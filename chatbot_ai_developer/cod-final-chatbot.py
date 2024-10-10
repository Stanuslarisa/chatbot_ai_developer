import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Configuration
load_dotenv(override=True)  # take environment variables from .env file
AZURE_OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
AZURE_SEARCH_API_ENDPOINT = os.getenv("AZURE_SEARCH_API_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

AZURE_OPENAI_API_VERSION = "2024-06-01"
index_name = 'langchain-vector-demo'

# Create embeddings and vector store instances
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_API_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

vector_store = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_API_ENDPOINT,
    azure_search_key=AZURE_SEARCH_API_KEY,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"retry_total": 4},
)

# Create a Langchain LLM using Azure Open AI
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_API_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT_NAME,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0,
    max_tokens=800,
    timeout=None,
    max_retries=2
)

# Creating a Langchain RAG chain for answering questions
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector_store.as_retriever(search_type="similarity", k=3, score_threshold=0.80)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit App
st.set_page_config(page_title="🦜🔗 Demo Streamlit App")
st.title('🦜🔗 Demo Streamlit App')

with st.form('my_form'):
    text = st.text_area('Întrebare:', '')
    submitted = st.form_submit_button('Trimite')
    if submitted and text:
        # Perform retrieval
        docs_and_scores = vector_store.similarity_search_with_relevance_scores(query=text, k=3, score_threshold=0.80)
        
        # Get response from the RAG chain
        response = rag_chain.invoke({"input": text})
        
        # Display the response and sources
        st.info(response["answer"])
        st.write("Surse:")
        for doc, score in docs_and_scores:
            st.write(f"- {doc.page_content} (Relevanță: {score})")  # Modificare aici
