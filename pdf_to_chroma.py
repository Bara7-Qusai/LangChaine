import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

# Load PDF document
loader = PyPDFLoader("rag_vs_fine_tuning.pdf")
data = loader.load()

# Split the document into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_documents(data)

# Create a Chroma vector database
embedding_function = OpenAIEmbeddings(api_key="OPENAI_API_KEY", model="text-embedding-3-small")

persist_directory = "./chroma_db"
vectorstore = Chroma.from_documents(docs, embedding=embedding_function, persist_directory=persist_directory)

# Save the database
vectorstore.persist()

# Configure the retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Test retrieval
query = "What is the difference between RAG and Fine-tuning?"
retrieved_docs = retriever.get_relevant_documents(query)

# Print retrieved documents
for i, doc in enumerate(retrieved_docs):
    print(f"\n--- Document {i+1} ---\n")
    print(doc.page_content)
