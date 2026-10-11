import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import xml.etree.ElementTree as ET

from hdev_loader import HDevLoader

# =====================================================================
# 1. LOAD YOUR FOLDER OF PDFS (OR A SAMPLE FILE)
# =====================================================================

# PDFS
PDF_FOLDER_PATH = os.path.expanduser("~/Downloads/halcon_pdfs") # Make sure this folder exists with PDFs inside

print(f"Loading documents from {PDF_FOLDER_PATH}...")
if os.path.exists(PDF_FOLDER_PATH):
    loaderpdf = DirectoryLoader(
        path=PDF_FOLDER_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader
    )
    docs_pdf = loaderpdf.load()
else:
    # Fallback to dummy document if folder isn't created yet
    from langchain_core.documents import Document

    docs_pdf = [
        Document(
            page_content="HALCON is a comprehensive standard software for machine vision."
        )
    ]

# HDEV

loaderhdev = DirectoryLoader(
    "/home/arzakuser/MVTec/HALCON-24.11-Progress-Steady/examples/hdevelop",
    glob="**/*.hdev",
    loader_cls=HDevLoader,
    # loader_kwargs={"encoding": "utf-8"},
)
docs_hdev = loaderhdev.load()


print(f"Loaded {len(docs_pdf)} document pages.")

# =====================================================================
# 2. CHUNK THE DOCUMENTS
# =====================================================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150
)
print(f"Loaded {len(docs_pdf)} total pages. Filtering out Table of Contents...")

# Filter out pages that look like a Table of Contents (lots of dot leaders)
clean_docs = []
for doc in docs_pdf:
    # If a page has multiple dot sequences like '. . . .', skip it
    if doc.page_content.count(". . .") > 3:
        continue
    # Skip pages explicitly titled as a Table of Contents at the very top
    if "Contents" in doc.page_content[:200] and "1 Overview" not in doc.page_content:
        continue
    clean_docs.append(doc)

print(f"Kept {len(clean_docs)} content pages after filtering.")

# Then pass 'clean_docs' into your text splitter instead of 'docs'
chunks = text_splitter.split_documents(clean_docs)
print(f"Split into {len(chunks)} chunks.")
chunks += docs_hdev

# =====================================================================
# 3. INITIALIZE EMBEDDINGS & VECTOR STORE
# =====================================================================
print("Generating embeddings and saving to ChromaDB...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma.from_documents(
    documents=chunks, embedding=embeddings, persist_directory="./chroma_db"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 15})

# =====================================================================
# 4. BUILD THE RAG CHAIN USING CORE RUNNABLES
# =====================================================================
llm = ChatOllama(model="llama3", temperature=0,
        num_ctx=16384,  # Allows the model to read large combined context (PDFs + whole .hdev files)
        num_predict=2048,  # Allows the model to output a full, uninterrupted script without cutting off
)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

def format_docs(docs):
    formatted = []
    for doc in docs:
        # Extract filename from path and get page number (if available)
        source_path = doc.metadata.get("source", "Unknown file")
        filename = os.path.basename(source_path)
        page = doc.metadata.get("page", "Unknown page")

        # Prepend citation metadata to the chunk text
        if doc.metadata.get("doc_type") == 'code':
            chunk_text = (
                f"[Source Code File: {filename}]\n{doc.page_content}"
            )
        else:
            chunk_text = (
                f"[Source PDF: {filename} | Page: {int(page) + 1}]\n{doc.page_content}"
            )
        formatted.append(chunk_text)

    return "\n\n".join(formatted)


# Modern LCEL RAG chain (replaces deprecated chain helpers)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# =====================================================================
# 5. QUERY THE SYSTEM
# =====================================================================
if __name__ == "__main__":
    # query = "What is HDevelop?"
    # print(f"\nQuerying: {query}\n" + "-" * 50)
    #
    # response = rag_chain.invoke(query)
    # print("\nAnswer:\n", response)
    print("\n" + "=" * 60)
    print("🚀 HALCON RAG System Ready!")
    print("Type your question below, or type 'quit' / 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            query = input("Question: ").strip()

            # Check for exit commands
            if query.lower() in ["quit", "exit", "q"]:
                print("Exiting RAG system. Goodbye!")
                break

            if not query:
                continue

            print("\nSearching and generating answer...")
            response = rag_chain.invoke(query)

            print("\nAnswer:")
            print(response)
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nExiting RAG system. Goodbye!")
            break