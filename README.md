# 🤖 MLOps Events Agent

> **An AI-powered semantic search and analysis platform for MLOps conference talks, built with ApertureDB, LangGraph, and Gemini 2.5 Pro**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://adb-query-agent.streamlit.app/)
[![Built with](https://img.shields.io/badge/Built_with-ApertureDB-00A8E8?style=for-the-badge)](https://www.aperturedata.io/)
[![Framework](https://img.shields.io/badge/Framework-LangChain-green?style=for-the-badge)](https://www.langchain.com/langgraph)

---

## 📖 Overview

This project demonstrates the power of combining **multi-modal AI-native databases** with **agentic AI systems** to create an intelligent search and analysis platform. Built on MLOps Conference talks, the system enables natural language queries, semantic search across text and video content, speaker analytics, and trend analysis.

**Key Technologies:**
- **ApertureDB**: Multi-modal database for storing talks, transcripts, videos, and embeddings
- **LangGraph**: ReAct agent framework for intelligent query orchestration
- **Gemini 2.5 Pro**: LLM for natural language understanding and reasoning
- **Streamlit**: Interactive web interface
- **Twelve Labs Marengo**: Video embeddings for semantic video search

---

## Features

### Core Capabilities

- **Natural Language Queries**: Ask questions in plain English about MLOps talks
- **Semantic Search**: Find talks by meaning, not just keywords
  - Text-based search (transcripts, abstracts, speaker bios)
  - Video-based search (visual + audio content understanding)
- **Speaker Analytics**: Analyze speaker activity, company representation, topic expertise
- **Trend Analysis**: Discover popular tools, technologies, and topics across conferences
- **Video Browser**: Browse and watch conference talk videos directly in the app
- **Intelligent Filtering**: Combine semantic search with metadata filters (dates, categories, speakers)

### Example Queries

```
"Which talks discuss AI agents with memory?"
"Show me the most popular talks from 2024"
"Who are the top 10 most active speakers?"
"What tools are trending in MLOps?"
"Find experts in vector databases and RAG"
```

---

## Project Architecture

### Data Layer (ApertureDB)

**Dataset**: 280 unique MLOps conference talks with:
- Talk metadata (title, speaker, company, abstract, keywords, categories)
- YouTube videos and URLs
- Enriched metadata (views, published dates, durations)
- Video transcripts (fetched via Apify YouTube scraper)

**Storage Model**:
```
Talk Entities (280)
  ├── Properties: talk_id, speaker_name, company_name, youtube_url, yt_views, etc.
  └── Connections:
      ├── Person entities (via TalkHasSpeaker edges)
      ├── Transcript chunks (via TalkHasTranscriptChunk)
      ├── Talk metadata (via TalkHasMeta)
      └── Speaker bios (via TalkHasSpeakerBio)

Video Entities (280)
  ├── Video blobs (MP4 files)
  └── Video embeddings (1024-dim, Twelve Labs Marengo)

Descriptor Sets (Vector Indexes):
  ├── ds_transcript_chunks_v1 (768-dim, chunked transcripts)
  ├── ds_talk_meta_v1 (768-dim, aggregated talk metadata)
  ├── ds_speaker_bio_v1 (768-dim, speaker information)
  └── marengo_2_7 (1024-dim, video embeddings)
```

### Agent Layer (LangGraph)

**7 Comprehensive Tools**:
1. `search_talks_by_filters` - Metadata filtering (dates, views, speakers, companies)
2. `search_talks_semantically` - Semantic search across text (transcripts/abstracts/bios)
3. `analyze_speaker_activity` - Speaker analytics and company breakdown
4. `get_talk_details` - Detailed talk information with transcripts
5. `find_similar_content` - Content recommendation engine
6. `analyze_topics_and_trends` - Trend analysis (tools, topics, technologies)
7. `get_unique_values` - Discover available values (events, categories, tracks)
8. `search_videos_semantically` - Semantic video search (visual + audio) (This isn't provided to the agent as a tool, but is available separately to manually search videos semantically)

**Agent Design**:
- **Pattern**: ReAct (Reasoning + Acting) using LangGraph's `create_react_agent`
- **Model**: Gemini 2.5 Pro (temperature=0.7)
- **Prompt**: Comprehensive system prompt with 14 few-shot examples demonstrating tool usage
- **Behavior**: Autonomously selects and chains tools based on user queries

### Frontend Layer (Streamlit)

**Multi-page Application**:
- **Main Page**: 
  - Chat with Agent (natural language interface)
  - Video Semantic Search (direct video search)
- **Video Browser Page**: Browse and watch all conference videos

**UI Features**:
- Real-time agent execution visibility
- Curated example queries across 4 categories
- Results sidebar with YouTube thumbnails
- Video players with fallback options
- Session-level connection pooling for performance

---

## How It Was Built

### Phase 1: Data Pipeline (Notebooks)

#### 1.1 Data Cleaning (`data_clean_adb.ipynb`)
- Started with raw MLOps events CSV (280 talks)
- Cleaned and normalized fields (speaker names, companies, keywords)
- Enriched with YouTube metadata via Apify scraper:
  - Video transcripts
  - View counts
  - Published dates
  - Video durations
- Generated unique `talk_id` for each talk

#### 1.2 Data Ingestion (`mlops_adb_ingest_data.ipynb`)
- Ingested Talk entities into ApertureDB with all metadata
- Created Person entities and TalkHasSpeaker edges
- Validated data integrity and relationships

#### 1.3 Text Embeddings (`mlops_adb_embeddings.ipynb`)
- **Model**: `google/embeddinggemma-300m` (768 dimensions)
- **Chunking Strategy**: 1000 chars with 200 char overlap
- Generated embeddings for:
  1. **Transcript chunks**: Each chunk linked to parent talk
  2. **Talk metadata**: Concatenated (title + abstract + learning outcomes + keywords)
  3. **Speaker bios**: Concatenated (job title + bio)
- Created descriptor sets in ApertureDB with proper indexing

#### 1.4 Video Embeddings (`adb-youtube.ipynb`)
- **Model**: Twelve Labs Marengo-retrieval-2.7 (1024 dimensions)
- Downloaded videos from YouTube URLs
- Generated video embeddings capturing visual + audio semantics
- Stored embeddings in `marengo_2_7` descriptor set
- Ingested video blobs into ApertureDB Video entities

### Phase 2: Query Development (`mlops_adb_queries.ipynb`)

Systematically tested various query patterns:
- Metadata filtering (dates, views, categories)
- k-NN semantic search across descriptor sets
- Multi-hop queries (talk → chunks → related talks)
- Constrained semantic search (within specific talks/speakers)
- Grouped results and aggregations

**Key Insight**: Identified common query patterns that users would need, which informed tool design.

### Phase 3: Tool Development (`tools/`)

Wrapped curated queries into **7 comprehensive, parameterized tools**:

Each tool follows a consistent pattern:
```python
@tool("tool_name", args_schema=InputSchema)
def tool_function(param1, param2, ...):
    """
    Comprehensive docstring explaining:
    - What the tool does
    - When to use it
    - Parameters and their meanings
    - Return structure
    """
    # Query construction
    # ApertureDB execution
    # Result formatting
    return structured_response
```

**Design Principles**:
- Self-contained (minimal agent coordination needed)
- Configurable via parameters (filters, limits, thresholds)
- Structured outputs (consistent JSON format)
- LangChain tool schemas (Pydantic validation)
- Comprehensive docstrings (LLM can understand usage)

### Phase 4: Agent Development (`agent/`)

#### 4.1 System Prompt (`agent/prompt.py`)
Created a **comprehensive 500+ line system prompt** including:
- Database schema explanation
- Tool selection guidelines
- **14 few-shot examples** covering:
  - Metadata filtering
  - Semantic search (text)
  - Speaker analysis
  - Talk deep dives
  - Content recommendations
  - Trend analysis
  - Value discovery
- Response guidelines and best practices

**Why Few-Shot Examples Matter**: They teach the agent *how* to use tools effectively, including parameter selection and query chaining.

#### 4.2 Agent Implementation (`agent/agent.py`)
- Used LangGraph's `create_react_agent` (prebuilt ReAct pattern)
- Configured Gemini 2.5 Pro with optimal temperature (0.7)
- Connected all 7 tools to the agent
- Implemented streaming for real-time execution visibility

### Phase 5: UI Development (`streamlit_app.py` + `ui/`)

Built a clean, modular Streamlit interface with:
- **Frontend/Backend Separation**: UI layer has zero database logic
- **Session Management**: Connection pooling for performance
- **Real-time Feedback**: Dynamic status messages during agent execution
- **Results Presentation**: Sidebar cards with YouTube thumbnails
- **Video Integration**: In-app video players with fallback options

### Phase 6: Video Search Integration

Extended the system with semantic video search:
- Added Twelve Labs client integration
- Created `search_videos_semantically` tool
- Built dedicated video search UI tab
- Enables queries like "find talks with live coding demos" (visual understanding)

---

## User Interface

### Chat with Agent
Natural language interface for querying the database:
- 8 curated example queries across 4 categories
- Real-time agent execution visibility
- Results displayed with YouTube links and metadata

### Video Semantic Search
Direct video search using Twelve Labs embeddings:
- Understands visual + audio content
- Searches beyond what's said (presentation style, visuals, demos)
- Configurable results with optional video players

### Video Browser
Browse and watch all 280 conference talks:
- Searchable dropdown by talk title
- Video metadata (FPS, duration, resolution)
- HTML5 video player with controls

---

## Technical Highlights

### Multi-Modal Semantic Search
Combines three types of embeddings for comprehensive search:
- **Text embeddings** (768-dim): Transcripts, abstracts, bios
- **Video embeddings** (1024-dim): Visual + audio content understanding
- **Hybrid approach**: Can search across both modalities

### Intelligent Query Orchestration
Agent can autonomously:
- Select single tools (simple queries)
- Chain multiple tools (complex analysis)
- Apply filters after semantic search
- Provide natural language summaries

### Performance Optimizations
- **Session-level caching**: Database connections, embedding models, clients
- **Lazy initialization**: Resources created on first use
- **Connection pooling**: One DB connection per user session

---

## 📊Dataset Statistics

- **Total Talks**: 280 unique conference presentations
- **Events**: MLOps World, GenAI World (2023-2024)
- **Speakers**: 200+ industry experts
- **Companies**: Google, Microsoft, Meta, Databricks, and 100+ more
- **Categories**: MLOps, Deployment, GenAI, Data Quality, Model Management, etc.
- **Total Transcript Chunks**: ~8,000 (chunked for semantic search)
- **Video Content**: ~280 hours of conference videos

---

## Use Cases

### For Researchers
- "Find all talks about LLM deployment strategies"
- "Show me research on model monitoring in production"
- "Which talks discuss RAG implementations?"

### For Practitioners
- "What tools are trending for MLOps in 2024?"
- "Find experts in feature engineering"
- "Show me talks from Databricks engineers"

### For Event Organizers
- "Who are our most active speakers?"
- "Which companies presented the most?"
- "What topics were most popular in 2024?"

### For Content Discovery
- "Find talks similar to this LangChain presentation"
- "Show me beginner-friendly talks about GenAI"
- "Recommend talks based on my interest in agents"

---

## 🏛️ Architecture Principles

### Clean Separation of Concerns
```
Frontend (UI)          → User interaction, display logic
  ↓
Agent (LangGraph)      → Reasoning, tool selection
  ↓
Tools (Functions)      → Query execution, result formatting
  ↓
Database (ApertureDB)  → Data storage, vector search
```

### Modularity
- Each tool is self-contained and testable
- UI components are reusable
- Agent is swappable (could use different LLM)
- Database queries are parameterized

### Scalability Considerations
- Vector indexes for fast k-NN search
- Session-level caching for performance
- Configurable result limits
- Async-ready architecture (could add async in future)

---

## 📁 Project Structure

```
.
├── agent/                      # LangGraph agent implementation
│   ├── agent.py               # ReAct agent with Gemini 2.5 Pro
│   ├── config.py              # Model settings, API keys
│   └── prompt.py              # System prompt + few-shot examples
│
├── tools/                      # 8 comprehensive query tools
│   ├── search_talks_by_filters.py
│   ├── search_talks_semantically.py
│   ├── analyze_speaker_activity.py
│   ├── get_talk_details.py
│   ├── find_similar_content.py
│   ├── analyze_topics_and_trends.py
│   ├── get_unique_values.py
│   ├── search_videos_semantically.py
│   └── utils.py               # Shared utilities
│
├── ui/                         # Streamlit UI components
│   ├── components.py          # Chat interface, agent display
│   ├── state.py               # Session state management
│   ├── sidebar.py             # Results display
│   ├── video_search.py        # Video search tab
│   ├── examples.py            # Curated example queries
│   └── styles.py              # Custom CSS
│
├── pages/                      # Streamlit multi-page app
│   └── 1_🎥_Video_Browser.py  # Video browsing page
│
├── notebooks/                  # Development notebooks
│   ├── data_clean_adb.ipynb   # Data cleaning
│   ├── mlops_adb_ingest_data.ipynb  # Data ingestion
│   ├── mlops_adb_embeddings.ipynb   # Text embeddings
│   ├── adb-youtube.ipynb      # Video embeddings
│   └── mlops_adb_queries.ipynb      # Query testing
│
├── data/                       # Dataset files
│   └── mlops-events-enriched.csv
│
│
├── streamlit_app.py           # Main Streamlit entry point
├── app.py                     # CLI interface
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

---

## Technology Stack

### Backend
- **ApertureDB**: Multi-modal database with vector search
- **LangGraph**: Agent orchestration framework
- **LangChain**: Tool integration and abstractions
- **Gemini 2.5 Pro**: LLM for reasoning and NLU
- **Sentence Transformers**: Text embeddings (`embeddinggemma-300m`)
- **Twelve Labs**: Video embeddings (Marengo-retrieval-2.7)

### Frontend
- **Streamlit**: Web application framework
- **Python 3.11+**: Core language

### Infrastructure
- **Streamlit Cloud**: Hosting and deployment
- **Environment Management**: `.env` + Streamlit secrets

---

## Key Learnings & Insights

### 1. Multi-Modal Embeddings Complement Each Other
Text embeddings excel at keyword-based semantic search, while video embeddings capture visual context and presentation style. Together, they provide comprehensive search capabilities.

### 2. Few-Shot Prompting is Critical for Tool Usage
The agent's performance dramatically improved with detailed few-shot examples showing parameter selection and tool chaining patterns.

### 3. Tool Design Matters
Self-contained tools that handle complete workflows (e.g., filter + search + format) reduce the need for complex agent coordination and improve reliability.

### 4. Session-Level Caching Significantly Improves Performance
Reusing database connections and embedding models across queries in the same session reduces latency from ~3s to <1s per query.

### 5. Cross-Version Compatibility Requires Defensive Coding
Supporting both legacy and modern LangChain content formats ensures the app works across different deployment environments.

