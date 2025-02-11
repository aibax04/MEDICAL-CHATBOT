from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from langchain.vectorstores import Pinecone as PineconeVectorStore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = download_hugging_face_embeddings()



# 🔹 Initialize Pinecone Client
pc = Pinecone(api_key="pcsk_5ftJtk_RPGwskkYfaFGx21SBduyqwS44EGQ2stdCX3rJRdhB1s32dos2WfAJMH6hRDhq3B")  # Replace with your API Key





# 🔹 Step 2: Define Index Name
index_name = "medicalchatbot"

# 🔹 Step 3: Get the Host of the Index
index_info = pc.describe_index(index_name)
host_url = "https://medicalchatbot-kvk4kva.svc.aped-4627-b74a.pinecone.io"  # Extract host URL

# 🔹 Step 4: Connect to Pinecone Index with Host
index = pc.Index(index_name, host=host_url)

print(f"✅ Connected to Pinecone index: {index_name}")

index_name = "medicalchatbot"

#Loading the index

index = pc.Index(index_name)
print(f"Connected to index: {index_name}")



PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})



embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
query_text = "what are heaaches?"


 # Convert to list for Pinecone


from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain.embeddings import HuggingFaceEmbeddings

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
from langchain.embeddings import HuggingFaceEmbeddings

# 🔹 Load the Sentence Transformer model for embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Convert a sample text into an embedding vector
# text = "How does the heart function?"
# query_embedding = embedding_model.embed_query(text)


# query_result = index.query(
#     vector=query_embedding,
#     top_k=5,  # More results
#     include_metadata=True
# )

# print("\n🔍 Query Results:")
# for match in query_result["matches"]:
#     score = match["score"]
#     text = match["metadata"].get("text", "⚠ No matching text found")
#     print(f"🔹 Confidence Score: {score:.4f}")
#     print(f"📄 Retrieved Text: {text}\n")



# 🔹 Connect to Pinecone Index
index = pc.Index(index_name)

from flask import Flask, request, jsonify, render_template
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# 🔹 Step 1: Initialize Pinecone Client
PINECONE_API_KEY = "pcsk_5ftJtk_RPGwskkYfaFGx21SBduyqwS44EGQ2stdCX3rJRdhB1s32dos2WfAJMH6hRDhq3B"  # Replace with your actual API Key
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot"
index = pc.Index(index_name)



# 🔹 Step 2: Load Embedding Model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🔹 Step 3: Render the Frontend Page
@app.route("/")
def home():
    return render_template("chat.html")  # Ensure you have `chat.html` in `templates/`

# 🔹 Step 4: Handle Chatbot Queries
@app.route("/get", methods=["POST"])
def chat():
    user_input = request.form.get("msg")  # Get user message from frontend
    print(f"📝 User Input: {user_input}")

    if not user_input:
        return jsonify({"response": "⚠ Error: No input received"}), 400  # Handle empty input

    # Convert user query into a vector embedding
    query_embedding = embedding_model.encode(user_input).tolist()

    # Query Pinecone for the most relevant result
    query_result = index.query(
        vector=query_embedding, 
        top_k=3,  
        include_metadata=True  
    )

    # Extract the best matching response
    if not query_result.get("matches", []):
        response_text = "❌ No relevant information found."
    else:
        response_text = query_result["matches"][0]["metadata"].get("text", "Sorry I don't have any access to external data , please clarify your question.")

    print(f"✅ Bot Response: {response_text}")  # Debugging log

    return jsonify({"response": response_text})  # ✅ Ensure Flask returns JSON response



