import openai
import faiss
import numpy as np
from sklearn.preprocessing import normalize
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
import os

# Ensure OpenAI API key is securely set up
openai.api_key = os.getenv('OPENAI_API_KEY')  # Use environment variable for security

# Sample documents for knowledge base
docs = [
    "Large language models (LLMs) are a type of machine learning model designed to understand and generate human language.",
    "FAISS is a library for efficient similarity search and clustering of dense vectors.",
    "OpenAI's GPT-4 is one of the most advanced language models, capable of tasks like translation, summarization, and question answering.",
    "Embeddings convert text data into a numerical format that can be used for tasks like semantic search and clustering."
]

# Step 1: Generate embeddings for the documents using OpenAI's Embeddings API
embedding_model = OpenAIEmbeddings(api_key=openai.api_key, model="text-embedding-ada-002")

# Generate embeddings for the documents
doc_embeddings = embedding_model.embed_documents(docs)

# Step 2: Normalize and create FAISS index for the embeddings
doc_embeddings = normalize(np.array(doc_embeddings))

# Create FAISS index using L2 distance
dimension = doc_embeddings.shape[1]  # Dimensionality of the embeddings
index = faiss.IndexFlatL2(dimension)  # Using L2 distance for similarity search
index.add(doc_embeddings.astype(np.float32))

# Step 3: Set up FAISS vector store
vectorstore = FAISS(index, embedding_model)

# Step 4: Define prompt template for the LLM
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Answer the following question using the provided context:\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:")
])

# Step 5: Define the LLM (e.g., GPT-4)
llm = ChatOpenAI(api_key=openai.api_key, model="gpt-4")

# Step 6: Function to retrieve relevant documents and generate an answer
def get_answer(query):
    # Step 6.1: Search for the top 3 relevant documents using FAISS
    query_embedding = embedding_model.embed_query(query)
    query_embedding = np.array(query_embedding).reshape(1, -1)
    query_embedding = normalize(query_embedding)
    
    # Retrieve top 3 similar documents based on cosine similarity
    _, indices = index.search(query_embedding.astype(np.float32), k=3)
    relevant_docs = [docs[i] for i in indices[0]]

    # Step 6.2: Format the prompt with the retrieved context and the user's query
    context = "\n".join(relevant_docs)
    prompt = prompt_template.format_messages(context=context, question=query)

    # Step 6.3: Generate the answer using GPT-4
    response = llm.invoke(prompt)

    return response['choices'][0]['message']['content']

# Step 7: Test the application with a sample question
question = "What is FAISS used for?"
answer = get_answer(question)
print(f"Answer: {answer}")
