from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
import streamlit as st

st.set_page_config(
    page_title="Local RAG Assistant", page_icon="🤖", layout="wide"
)

st.title("👁️Local RAG Developer Assistant")
st.caption(
    "Offline RAG powered by Ollama (llama3), ChromaDB, and your local .hdev"
    " scripts & manuals."
)


# 1. Initialize RAG components (Cached so they only load once)
@st.cache_resource
def load_rag_system():
  embeddings = OllamaEmbeddings(model="nomic-embed-text")
  vector_store = Chroma(
      persist_directory="./chroma_db", embedding_function=embeddings
  )
  retriever = vector_store.as_retriever(
      search_kwargs={"k": 15}
  )  # Pulls top 3 matches (code or docs)
  llm = ChatOllama(model="llama3", num_ctx=16384, num_predict=2048)
  return retriever, llm


retriever, llm = load_rag_system()


# 2. Format documents function with your custom metadata handling
def format_docs(docs):
  formatted = []
  for doc in docs:
    doc_type = doc.metadata.get("doc_type", "documentation")

    if doc_type == "code":
      filename = doc.metadata.get("source", "unknown_script.hdev")
      header = f"=== HALCON CODE EXAMPLE: {filename} ==="
    else:
      source = doc.metadata.get("source", "Unknown").split("/")[-1]
      page = doc.metadata.get("page", 0)
      page_display = int(page) + 1 if str(page).isdigit() else page
      header = f"=== MANUAL: {source} | Page: {page_display} ==="

    formatted.append(f"{header}\n{doc.page_content}")

  return "\n\n\n".join(formatted)


# 3. Initialize chat history in session state
if "messages" not in st.session_state:
  st.session_state.messages = []

# Display prior chat history
for message in st.session_state.messages:
  with st.chat_message(message["role"]):
    st.markdown(message["content"])
    if "sources" in message and message["sources"]:
      with st.expander("🔍 View Retrieved Sources & Context"):
        st.markdown(message["sources"])

# 4. Handle user input
if query := st.chat_input("Ask for a HALCON operator, code, or concept..."):
  # Append user message
  st.session_state.messages.append({"role": "user", "content": query})
  with st.chat_message("user"):
    st.markdown(query)

  # Generate response
  with st.chat_message("assistant"):
    with st.spinner("Searching local code repo and manuals..."):
        # Retrieve relevant docs
      retrieved_docs = retriever.invoke(query)
      context = format_docs(retrieved_docs)

      # Construct prompt with strict context constraint
      prompt = f"""You are an expert HALCON developer. Answer the user's question using the provided local code examples and documentation context. If the user asks for code, generate a clean HALCON script based on the operator patterns and context found below.

--- CONTEXT ---
{context}

--- QUESTION ---
{query}
"""

      # Stream the response from Ollama
      response_placeholder = st.empty()
      full_response = ""

      for chunk in llm.stream(prompt):
        full_response += chunk.content
        response_placeholder.markdown(full_response + "▌")

      response_placeholder.markdown(full_response)

      # Show sources used in an expander
      sources_text = ""
      if retrieved_docs:
        sources_text = f"```text\n{context}\n```"
        with st.expander("🔍 View Retrieved Sources & Context"):
          st.markdown(sources_text)

      # Save assistant response to history
      st.session_state.messages.append({
          "role": "assistant",
          "content": full_response,
          "sources": sources_text,
      })