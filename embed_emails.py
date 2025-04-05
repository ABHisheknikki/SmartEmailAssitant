import os
import json
import shutil
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Directories
SOURCE_EMAIL_DIR = "source_emails"
DUMP_EMAIL_DIR = "dump_emails"
DB_EMAILS_DIR = "db_emails"

# Ensure directories exist
os.makedirs(SOURCE_EMAIL_DIR, exist_ok=True)
os.makedirs(DUMP_EMAIL_DIR, exist_ok=True)
os.makedirs(DB_EMAILS_DIR, exist_ok=True)

# Email schema
class Email(BaseModel):
    threadId: str
    messageId: str
    to: str
    from_: str = Field(..., alias="from")
    subject: str
    body: str
    time: str
    date: str

# Initialize embedder
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=api_key
)

# Text splitter
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Initialize Chroma DB
vectorstore = Chroma(
    persist_directory=DB_EMAILS_DIR,
    embedding_function=embedding_model
)

# Process JSON email files
for filename in os.listdir(SOURCE_EMAIL_DIR):
    if filename.endswith(".json"):
        filepath = os.path.join(SOURCE_EMAIL_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                emails_json = json.load(f)
                emails = [Email(**email) for email in emails_json]
            except Exception as e:
                print(f"❌ Error parsing {filename}: {e}")
                continue

        all_docs = []
        for email in emails:
            full_text = f"""Subject: {email.subject}
From: {email.from_}
To: {email.to}
Date: {email.date} {email.time}

{email.body}"""
            chunks = splitter.create_documents([full_text])
            for chunk in chunks:
                chunk.metadata = {
                    "threadId": email.threadId,
                    "messageId": email.messageId,
                    "from": email.from_,
                    "to": email.to,
                    "subject": email.subject,
                    "date": email.date
                }
            all_docs.extend(chunks)

        # Store all chunks into Chroma
        vectorstore.add_documents(all_docs)
        vectorstore.persist()

        print(f"✅ Embedded and stored emails from: {filename}")

        # Move file to dump
        shutil.move(filepath, os.path.join(DUMP_EMAIL_DIR, filename))

print("📬 All emails embedded and stored in 'db_emails'.")
