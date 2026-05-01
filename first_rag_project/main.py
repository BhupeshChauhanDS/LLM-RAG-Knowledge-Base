import os
import langchain

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI



# SETTINGS

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
langchain.debug = False                                                                # Manually setting the missing attribute
langchain.verbose = False
langchain.llm_cache = None                                                             # This fixes the new error you just got 



# STEP 1 LOAD PDF

loader = PyPDFLoader("Data/sample.pdf")
documents = loader.load()

# STEP 2 SPLIT TEXT INTO CHUNKS

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

#  Print chunks

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i+1} ---\n")
    print(chunk.page_content)

# STEP 3  CREATE EMBEDDINGS + FAISS VECTOR STORE


embeddings = GoogleGenerativeAIEmbeddings(
model="models/gemini-embedding-001", # Change this line
    google_api_key=api_key
)

vector_store = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
print("Vector store created successfully!")



# STEP 4 CREATE RETRIEVER

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

#  Question

query = "What skills does Bhupesh have?"

# Retrieve relevant chunks

results = retriever.invoke(query)

for i, doc in enumerate(results):
    print(f"\n--- Result {i+1} ---\n")
    print(doc.page_content)


#  STEP 5  INITIALIZE LLM (GEMINI)      


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=api_key,
    temperature=0.3
)

context = "\n\n".join([doc.page_content for doc in results])

# STEP 6 BUILD PROMPT + GET ANSWER

prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nFINAL ANSWER:\n")
print(response.content)

