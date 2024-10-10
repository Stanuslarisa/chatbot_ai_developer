# Demo 06: Cautarea in Azure AI Search folosind vector stores, si folosirea Azure OpenAI într-un scenariu de RAG pentru a răspunde la întrebări

index_name: str = 'langchain-vector-demo'

import os
from dotenv import load_dotenv
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
from pprint import pprint


# Configuration

load_dotenv(override=True)  # take environment variables from .env file
AZURE_OPENAI_API_ENDPOINT = os.getenv("AZURE_OPENAI_API_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME")
AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
AZURE_SEARCH_API_ENDPOINT = os.getenv("AZURE_SEARCH_API_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")

AZURE_OPENAI_API_VERSION = "2024-06-01"

# Create embeddings and vector store instances
embeddings: AzureOpenAIEmbeddings = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME,
    openai_api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_API_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
)

vector_store: AzureSearch = AzureSearch(
    azure_search_endpoint=AZURE_SEARCH_API_ENDPOINT,
    azure_search_key=AZURE_SEARCH_API_KEY,
    index_name=index_name,
    embedding_function=embeddings.embed_query,
    additional_search_client_options={"retry_total": 4},
)

# Perform a vector similarity search with relevance scores
# https://github.com/langchain-ai/langchain/blob/master/docs/docs/integrations/vectorstores/azuresearch.ipynb

docs_and_scores = vector_store.similarity_search_with_relevance_scores(
    query="Care este totul activelor in 2023 in ceea ce priveste grupul?",
    k=3,
    score_threshold=0.80,
)
from pprint import pprint
pprint(docs_and_scores)

# Create a Langchain LLM using Azure Open AI
# https://python.langchain.com/docs/integrations/chat/azure_chat_openai/

from langchain_openai import AzureChatOpenAI

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
# https://python.langchain.com/docs/tutorials/rag

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

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

response = rag_chain.invoke({"input": "Care este totul activelor in 2023 in ceea ce priveste grupul?"})
print(response["answer"])
