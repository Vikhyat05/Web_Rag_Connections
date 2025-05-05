# 🧠 ContinuOS – LLM OS for Knowledge Workers

**ContinuOS** is an AI-powered operating system for knowledge workers. It connects to personal or organizational knowledge bases and enhances them with real-time web data using Retrieval-Augmented Generation (RAG). 

This repository contains two core agents:

- `search_agent.py` – Intelligent query handler that retrieves and generates cited responses
- `connections_agent.py` – Insight generator that identifies meaningful links between topics

---

## 🧩 Project Components

| File | Description |
|------|-------------|
| `search_agent.py` | Streams structured, cited responses using LangChain agents, Pinecone, Brave Search, and ListenNotes |
| `connections_agent.py` | Analyzes topic relationships in the knowledge base and suggests novel insights |

---

## 🔍 Features

### `search_agent.py` – Search Agent

- RAG pipeline over **Pinecone** + **Brave Search** + **ListenNotes**
- Knowledge base-first answers with proper **citations**
- Enriches answers with **YouTube videos**, **articles**, and **podcasts**
- **Streaming GPT-4** output with structured markdown format
- Supports contextual chat memory and system-level formatting logic

### `connections_agent.py` – Connections Engine

- Uses **CSV-based tagged knowledge base**
- Extracts and analyzes **topic tags**
- Identifies connections between two user-defined concepts
- Surfaces related content and synthesizes **original insights**
- Includes citation of sources for every connection

---

## 📚 Use Cases

- Build a searchable OS on your notes, papers, or documents
- Explore deep relationships between concepts like “AI” and “Healthcare”
- Extend personal knowledge with real-time internet context
- Generate content, strategy, or learning paths from your own data

---

## ⚙️ Tech Stack

- **LangChain Agents + Tools**
- **OpenAI GPT-4** (function calling + streaming)
- **Pinecone** for vector retrieval
- **Brave Search API**
- **ListenNotes API**
- **Pandas**, **Pydantic**, **dotenv**

