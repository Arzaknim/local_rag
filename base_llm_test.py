from langchain_ollama import ChatOllama

# 1. Initialize the base LLM
llm = ChatOllama(model="llama3", num_predict=1024)

# 2. Send a query directly without RAG
query = "give me the code for gen measure arc in halcon"
print(f"=== SENDING RAW QUERY TO LLM: '{query}' ===\n")

response = llm.invoke(query)

print(response.content)