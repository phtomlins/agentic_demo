# embed_and_store.py
from chromadb import Client
from sentence_transformers import SentenceTransformer
from crewai.llms import OllamaLLM

# Initialize
client = Client(path="./chroma_db")
collection = client.create_collection(name="docs")

embedder = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight locally
llm = OllamaLLM(model="mistral")

# Embed and store
docs = ["Doc A text...", "Doc B text..."]
embeddings = embedder.encode(docs).tolist()
collection.add(documents=docs, embeddings=embeddings)

# Retrieval function (used inside an agent tool)
def retrieve_similar(query, top_k=3):
    q_emb = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=q_emb, n_results=top_k)
    return results['documents'][0]