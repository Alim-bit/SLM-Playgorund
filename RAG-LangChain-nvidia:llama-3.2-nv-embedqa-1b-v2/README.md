# RAG-LangChain-NVIDIA

This project implements a Retrieval-Augmented Generation (RAG) system using NVIDIA embeddings and LangChain. The system answers AWS Lambda-related questions using the AWS Lambda Developer Guide as its knowledge base.

## Features
- Embeds AWS Lambda documentation using NVIDIA's `llama-3.2-nv-embedqa-1b-v2` model.
- Stores embeddings in ChromaDB.
- Answers queries about AWS Lambda.

## License

This repository incorporates content from the [AWS Lambda Developer Guide](https://github.com/awsdocs/aws-lambda-developer-guide), which is licensed under the [Amazon Software License](https://aws.amazon.com/asl/). 

By using this repository, you agree to comply with the terms of the Amazon Software License as it pertains to the use of AWS documentation.

### Notes:
- All custom code and implementation in this repository are subject to the terms outlined below (choose your license, e.g., MIT, Apache 2.0).
- AWS documentation and related assets remain the property of Amazon Web Services and are licensed separately.
