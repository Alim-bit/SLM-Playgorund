import os
import numpy as np
import nltk
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from openai import OpenAI
import chromadb

# Load environment variables
load_dotenv()

# Download necessary NLTK packages
nltk.download('punkt')

# NVIDIA API key from environment variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

PROMPT_TEMPLATE = """ You are AWS Plugin 
Answer the question based only on the following context:

"{context}"

---

Answer the question based on the above context it should in first person: "{question}"
"""

# Step 1: Load AWS Lambda Documentation
def load_documents():
    print("Loading AWS Lambda documentation...")
    loader = DirectoryLoader(
        'aws-lambda-developer-guide',
        glob='**/*.md',
        recursive=True
    )
    documents = loader.load()

    # Split documents into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(documents)
    chunkdata = (f"Split {len(documents)} documents into {len(chunks)} chunks. {chunks}")

    with open("context.txt", "w") as f:
        f.write(chunkdata)

    return chunks

# Step 2: Generate Embeddings using NVIDIA
def generate_embeddings(docs):
    print("Generating embeddings using NVIDIA model...")
    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=NVIDIA_API_KEY,
        truncate="NONE"
    )
    return embeddings.embed_documents([doc.page_content for doc in docs])

# Step 3: Store Embeddings in ChromaDB
def store_embeddings(docs, embedded_docs):
    print("Storing embeddings in ChromaDB...")

    # Define the persist directory for ChromaDB
    persist_directory = "./chromadb_storage"

    # Initialize NVIDIA embeddings for retrieval
    embedding_function = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=NVIDIA_API_KEY,
        truncate="NONE"
    )

    # Initialize ChromaDB vector store
    vectorstore = Chroma(
        collection_name="aws_lambda_docs",
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

    # Add documents and their embeddings
    vectorstore.add_texts(
        texts=[doc.page_content for doc in docs],
        metadatas=[{"source": doc.metadata.get("source", "Unknown")} for doc in docs],
        ids=[str(idx) for idx in range(len(docs))]
    )

    # Persist the data
    vectorstore.persist()

    # Return the vectorstore as the retriever
    return vectorstore.as_retriever()


# Step 4: Query the RAG System and Use OpenRouter's Gemma
def answer_query(retriever, query):
    print(f"Querying the system with: {query}")
    
    # Retrieve relevant documents using the retriever
    relevant_docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in relevant_docs])

    # Use Gemma to generate a refined answer
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    prompt = PROMPT_TEMPLATE.format(context=context, question=query)
    with open("query_context.txt", "w") as f:
        f.write(prompt)

    completion = client.chat.completions.create(
        model="google/gemini-2.0-flash-exp:free",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Check if response is valid
    if completion and completion.choices and completion.choices[0].message.content:
        answer = completion.choices[0].message.content

        # Save the answer to `answer.txt`
        with open("answer.txt", "w") as f:
            f.write(answer)

        return answer
    else:
        raise ValueError("Invalid response from OpenRouter API")


# Main Workflow
if __name__ == "__main__":
    # Load documents
    docs = load_documents()

    
    embedded_docs = generate_embeddings(docs)
    os.makedirs("embeddings", exist_ok=True)
    np.save("embeddings/aws_lambda_embeddings.npy", embedded_docs)

    # Store embeddings in ChromaDB and get a retriever
    retriever = store_embeddings(docs, embedded_docs)

    # Test the RAG system
    query = "How do I package a Python application for AWS Lambda?"
    answer = answer_query(retriever, query)
    print(f"Answer:\n{answer}")
