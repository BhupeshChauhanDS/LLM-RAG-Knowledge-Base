# End-to-End RAG Pipeline using LangChain and Gemini

## Overview

A simple Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about your PDF documents using LangChain, Gemini API, and FAISS.

## Features

- PDF document loading
- Text chunking
- Embedding generation
- Vector database using FAISS
- Semantic search
- Question answering using Gemini LLM

## Tech Stack

- Python
- LangChain
- Gemini API
- FAISS

## Project Structure

```text
Data/
 └── sample.pdf

main.py
README.md
requirements.txt
```

## How It Works

1. **Load** — PDF is loaded and parsed using LangChain's document loader
2. **Chunk** — Text is split into smaller overlapping chunks
3. **Embed** — Each chunk is converted into a vector using Gemini embeddings
4. **Retrieve & Answer** — On a query, the most relevant chunks are fetched from FAISS and passed to Gemini LLM for an answer

## Example Question

What skills does Bhupesh have?.

## Demo Output

![RAG Output](screenshots/output.png)

## Limitations

- Only supports single PDF at a time
- Answer quality depends on chunk size and overlap settings
- Gemini API requires an active internet connection and valid key
