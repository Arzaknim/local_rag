from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text", base_url="http://localhost:11434"
)
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
results = db.similarity_search("something something", k=3)

for r in results:
  print(f"Source: {r.metadata.get('source')}")
  print(r.page_content[:200])
  print("-" * 40)