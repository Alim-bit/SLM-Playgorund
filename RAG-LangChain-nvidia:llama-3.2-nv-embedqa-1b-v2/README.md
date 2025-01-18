# RAG-LangChain-NVIDIA

This project implements a Retrieval-Augmented Generation (RAG) system using NVIDIA embeddings and LangChain. The system answers AWS Lambda-related questions using the AWS Lambda Developer Guide as its knowledge base.

## Features
- Embeds AWS Lambda documentation using NVIDIA's `llama-3.2-nv-embedqa-1b-v2` model.
- Stores embeddings in ChromaDB.
- Answers queries about AWS Lambda.

## Setup
1. Clone the AWS Lambda Developer Guide:
   ```bash
   git clone https://github.com/awsdocs/aws-lambda-developer-guide.git
