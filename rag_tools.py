from langchain.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

# Load document vectorstore
doc_vectorstore = Chroma(
    persist_directory="db",
    embedding_function=embedding_model
)
doc_retriever = doc_vectorstore.as_retriever(search_kwargs={"k": 4})

# Load email vectorstore
email_vectorstore = Chroma(
    persist_directory="db_emails",
    embedding_function=embedding_model
)
email_retriever = email_vectorstore.as_retriever(search_kwargs={"k": 4})

retrievers = {
    "doc_retriever": doc_retriever,
    "email_retriever": email_retriever
}
def retrieve_docs(state):
   
   
    query = state.get("query")

    if not query:
        return {"docs": []}
    docs = retrievers["doc_retriever"].invoke(query)
    return {"docs": docs}
def retrieve_emails(state):
    query = state.get("query")
    if not query:
        return {"emails": []}
    emails = retrievers["email_retriever"].invoke(query)
    return {"emails": emails}

