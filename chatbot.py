import os
import json
from dotenv import load_dotenv
from langgraph_flow import graph  # Make sure langgraph_flow.py exports `graph`

# Load environment variables
load_dotenv()

# Chat history file
HISTORY_FILE = "chat_history.json"

# Load existing history (if any)
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Save updated history
def save_history(history):
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# Start interaction
print("🤖 AI Email Assistant (RAG-powered by LangGraph)")
print("Type 'exit' to quit.\n")

chat_history = load_history()

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break

    try:
        # Run query through LangGraph
        result = graph.invoke({
    "query": user_input,
    "history": chat_history  # Pass history as part of state
})

        answer = result.get("answer", "No answer returned.")

        # Display assistant response
        print("🤖:", answer)

        # Save the current turn to history
        chat_history.append({"query": user_input, "answer": answer})
        save_history(chat_history)

    except Exception as e:
        print("❌ Error:", str(e))

