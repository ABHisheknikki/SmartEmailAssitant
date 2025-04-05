from langgraph.graph import END, StateGraph
from langchain.schema.runnable import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from rag_tools import retrieve_docs, retrieve_emails, retrievers

from dotenv import load_dotenv
from typing import TypedDict
import os

# 1. Load API key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2. Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.3
)

# 3. Define State Schema for LangGraph
class RAGState(TypedDict):
    query: str
    docs: list  # list of Documents
    emails: list  # list of Emails
    answer: str

# 4. Define stateful node functions
def retrieve_docs(state):
    query = state["query"]
    docs = retrievers["doc_retriever"].invoke(query)
    return {"docs": docs}

def retrieve_emails(state):
    query = state["query"]
    emails = retrievers["email_retriever"].invoke(query)
    return {"emails": emails}

def generate_response(state):
    query = state["query"]
    docs = state.get("docs", [])
    emails = state.get("emails", [])
    history = state.get("history", [])

    context = "\n\n".join([d.page_content for d in (docs + emails)])

    # Format chat history
    formatted_history = ""
    for turn in history[-5:]:  # Limit to last 5 turns
        formatted_history += f"User: {turn['query']}\nAI: {turn['answer']}\n"

    prompt = f"""You are an AI email assistant. Use the following email/document context and chat history to respond appropriately.

Chat History:
{formatted_history}

Context:
{context}

Current User Query:
{query}

Answer:"""

    return {"answer": llm.invoke(prompt).content}

# 5. Define LangGraph Flow with State Schema
workflow = StateGraph(RAGState)

workflow.add_node("retrieve_docs", RunnableLambda(retrieve_docs))
workflow.add_node("retrieve_emails", RunnableLambda(retrieve_emails))
workflow.add_node("generate_response", RunnableLambda(generate_response))

workflow.set_entry_point("retrieve_docs")
workflow.add_edge("retrieve_docs", "retrieve_emails")
workflow.add_edge("retrieve_emails", "generate_response")
workflow.add_edge("generate_response", END)

# 6. Compile the graph
graph = workflow.compile()

