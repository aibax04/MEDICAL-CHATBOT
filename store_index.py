
from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')

# print(PINECONE_API_KEY)
# print(PINECONE_API_ENV)

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
import fitz  # PyMuPDF for PDF text extraction
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Pinecone as PineconeVectorStore
from sentence_transformers import SentenceTransformer


# 🔹 Step 1: Extract Text from Uploaded PDF
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Path to the uploaded PDF
pdf_path = r"C:\Users\mohda\OneDrive\Desktop\MEDICAL-CHATBOT\data\Medical_book.pdf"  # Replace with the actual path

# Extract text from PDF
extracted_text = extract_text_from_pdf(pdf_path)

# 🔹 Step 2: Split Extracted Text into Smaller Chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_text(extracted_text)

print(f"✅ Extracted {len(text_chunks)} text chunks from PDF.")

from pinecone import Pinecone

# 🔹 Step 1: Initialize Pinecone Client
pc = Pinecone(api_key="pcsk_5ftJtk_RPGwskkYfaFGx21SBduyqwS44EGQ2stdCX3rJRdhB1s32dos2WfAJMH6hRDhq3B")  # Replace with your API key

# 🔹 Step 2: Define Index Name and Region
index_name = "medicalchatbot"
region = "us-east-1"  # Change this to match your Pinecone region


# Connect to the existing index
index = pc.Index(index_name)
print(f"Connected to index: {index_name}")

# 🔹 Step 3: Initialize Pinecone Client
pc = Pinecone(api_key="pcsk_5ftJtk_RPGwskkYfaFGx21SBduyqwS44EGQ2stdCX3rJRdhB1s32dos2WfAJMH6hRDhq3B")  # Replace with your actual API Key

# Define Pinecone index details
index_name = "medicalchatbot"
expected_dim = 384  # Ensure embeddings match this dimension



# Connect to the existing index
index = pc.Index(index_name)
print(f"Connected to index: {index_name}")

# 🔹 Step 4: Convert Text Chunks into Vector Embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = embedding_model.embed_documents(text_chunks)

# 🔹 Step 5: Format Data for Pinecone Upsert
vectors = [
    {"id": f"doc_{i}", "values": embeddings[i], "metadata": {"text": text_chunks[i]}}
    for i in range(len(text_chunks))
]

# 🔹 Step 6: Store Vectors in Pinecone
batch_size = 100  # To prevent exceeding Pinecone's request limits
for i in range(0, len(vectors), batch_size):
    index.upsert(vectors=vectors[i : i + batch_size])
    print(f"Upserted {i + len(vectors[i : i + batch_size])} vectors so far.")

print("✅ PDF text chunks successfully converted to vectors and stored in Pinecone!")

