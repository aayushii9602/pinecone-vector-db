# 📘 PDF Q&A Bot Using LangChain, OpenAI, and Pinecone

This project enables interactive querying of a PDF document using natural language. It leverages [LangChain](https://www.langchain.com/), [OpenAI](https://openai.com/), and [Pinecone](https://www.pinecone.io/) to load and embed a PDF, store its vector representations, and generate context-aware responses.

## 🚀 Features

- Loads and splits a PDF into manageable chunks.
- Embeds text chunks using OpenAI's Embedding API.
- Stores embeddings in Pinecone vector database.
- Retrieves relevant content using similarity search.
- Answers user questions based on context using OpenAI's GPT model.
- Interactive CLI interface.

## 🛠️ Prerequisites

- Node.js (v18+)
- OpenAI API key
- Pinecone API key and environment
- A PDF file 
