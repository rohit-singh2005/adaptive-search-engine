# 🔍 Adaptive Search Engine

> A personalized information retrieval system that learns from your behavior and delivers explainable, adaptive search results over the **MS MARCO** passage corpus.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-009688)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

Traditional search engines return the same results for every user, ignoring individual preferences and past behavior. This project implements a **retrieval system that adapts in real-time** using:

- 🧠 **User profile vectors** built from questionnaire answers and refined via relevance feedback
- ⚡ **Hybrid retrieval** — dense semantic search (FAISS + SentenceTransformers) blended with sparse BM25 keyword matching
- 🎯 **Multi-signal scoring** — query relevance, profile similarity, source preference, and recency
- 💬 **Explainable rankings** — every result shows *why* it was ranked highly
- 📊 **Live evaluation metrics** — P@5, P@10, MRR, Recall@10 tracked across feedback rounds

---

## 🖥️ Demo Screenshots

| Landing Page | Search Interface |
|---|---|
| User profile selection with feature overview | Personalized dual-panel results with explanations |

---

## 🏗️ System Architecture

```
┌──────────────────────┐     ┌──────────────────────────────┐     ┌────────────────┐
│  Streamlit UI        │────▶│     IREngine Core             │────▶│  FAISS Index   │
│  (app.py)            │◀────│   (ir_engine.py)              │◀────│ (IndexFlatIP)  │
│                      │     │                               │     └────────────────┘
│ • User Selection     │     │ • SentenceTransformer         │
│ • Questionnaire      │     │ • Topic Classification        │     ┌────────────────┐
│ • Sidebar Controls   │     │ • Source Detection             │────▶│  Disk Cache    │
│ • Search + Expansion │     │ • Mismatch Detection           │     │  (Parquet +    │
│ • Feedback (👍👎)    │     │ • Multi-Signal Scoring         │     │   FAISS .index)│
│ • Explainability     │     │ • BM25 Hybrid Scoring          │     └────────────────┘
│ • Analytics Panel    │     │ • Query Expansion (TF-IDF)     │
│ • Eval Metrics       │     │ • Profile Persistence (JSON)   │     ┌────────────────┐
│ • Search History     │     │ • Evaluation Metrics           │────▶│ User Profiles  │
│                      │     │ • Rocchio Feedback              │     │ (JSON on disk) │
└──────────────────────┘     └──────────────────────────────┘     └────────────────┘
```

### Component Summary

| Component | Technology | Purpose |
|---|---|---|
| Embedding Model | `all-MiniLM-L6-v2` (384-dim) | Dense vector encoding for passages and queries |
| Vector Index | FAISS `IndexFlatIP` | Exact cosine similarity search |
| Data Store | Parquet + FAISS `.index` | Disk caching for fast restarts |
| User Profiling | Rocchio Algorithm | Updates interest vectors from relevance feedback |
| Sparse Retrieval | BM25 Okapi (`rank_bm25`) | Keyword-based scoring blended with dense retrieval |
| Query Expansion | TF-IDF + profile vectors | Auto-expands queries with interest-relevant terms |
| Topic Classifier | Keyword-based | Classifies queries/passages into predefined topics |
| Source Detector | Keyword-based | Infers passage source type (official, blog, news, forum) |
| Explainability | Multi-factor reasoning | Human-readable explanations for each ranking |
| Frontend | Streamlit | Interactive UI with questionnaire, search, and feedback |

---

## ✨ Features

### 🎯 Personalization
- **Initial Questionnaire** — Users set topic interests, search depth, recency preference, source type, and topics to avoid before their first search
- **Rocchio Relevance Feedback** — Thumbs up/down on results updates the user's profile vector in real-time
- **Persistent Profiles** — Profiles, preferences, liked documents, and feedback logs saved to disk and restored on restart

### 🔎 Retrieval
- **Dense Retrieval** — SentenceTransformer embeddings + FAISS inner-product search
- **BM25 Hybrid** — BM25 Okapi sparse scoring blended with dense retrieval (user-adjustable slider)
- **Query Expansion** — Profile-driven TF-IDF expansion adds related terms to improve recall
- **Multi-Signal Scoring**:
  ```
  score = w_q × query_sim + w_p × profile_sim + w_s × source_boost + w_r × recency_boost
  ```

### ⚠️ Topic Mismatch Detection
- Detects when a query touches a topic the user prefers to avoid
- Reduces personalization weight and displays a conflict warning banner

### 💡 Explainability
Each ranked result includes a human-readable explanation:
```
· Strong match to your query (72% similarity)
· Aligns with your learned interests
· Similar to a document you liked (shared terms: python, machine, learning)
```

### 📊 Analytics & Metrics
- **Profile Strength Indicator** — Progress bar showing profile vector magnitude
- **Topic Affinity Chart** — Plotly bar chart of cosine similarity between user profile and each topic
- **Evaluation Metrics** — P@5, P@10, MRR, Recall@10 computed per session
- **Metric Trend Chart** — Line chart tracking P@5 and MRR across feedback rounds
- **Recent Search History** — Last 5 queries with timestamps

### 👥 Multi-User Support
- 5 independent user profiles with customizable display names
- Each user maintains: profile vector, preferences, liked/disliked document IDs, search history, feedback log

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/rohit-singh2005/adaptive-search-engine.git
cd adaptive-search-engine

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

> **First-run note:** On the very first launch the app downloads ~50,000 passages from MS MARCO via HuggingFace and builds the FAISS index. This takes a few minutes. All data is cached to disk — subsequent starts are instant.

---

## 📁 Project Structure

```
adaptive-search-engine/
│
├── app.py                  # Streamlit UI — all three screens (landing, questionnaire, search)
├── ir_engine.py            # Core IREngine class (indexing, search, feedback, metrics)
├── methodology.md          # Detailed technical methodology document
├── requirements.txt        # Python dependencies
│
├── data/                   # Auto-created on first run
│   ├── msmarco_50000.parquet   # Cached MS MARCO passages
│   ├── msmarco_50000.index     # Cached FAISS index
│   └── user_profiles.json      # Persistent user profiles & preferences
│
└── README.md
```

---

## 🔧 Configuration

| Parameter | Default | Location | Description |
|---|---|---|---|
| `subset_size` | `50000` | `app.py` line 431 | Number of MS MARCO passages to load |
| `model_name` | `all-MiniLM-L6-v2` | `IREngine.__init__` | SentenceTransformer model |
| `dense_weight` | `0.7` | Sidebar slider | Blend between dense and BM25 scoring |
| `personalization_weight` | `0.5` | Sidebar slider | Profile influence on ranking |
| Rocchio α / β / γ | `1.0 / 0.75 / 0.25` | `ir_engine.py` | Feedback update parameters |

---

## 🧪 Evaluation Metrics

After providing at least one piece of feedback (like/dislike), an evaluation panel appears with:

| Metric | Formula | Description |
|---|---|---|
| **P@5** | relevant_in_top5 / 5 | Precision at rank 5 |
| **P@10** | relevant_in_top10 / 10 | Precision at rank 10 |
| **MRR** | 1 / rank_of_first_relevant | Mean Reciprocal Rank |
| **Recall@10** | found_in_top10 / total_relevant | Recall at rank 10 |

Liked documents serve as the relevance ground truth. The trend chart shows how P@5 and MRR improve as you provide more feedback — demonstrating the adaptive improvement of the system.

---

## 📚 Methodology

See [`methodology.md`](./methodology.md) for a comprehensive technical write-up covering:
- Data ingestion and caching strategy
- Embedding and indexing pipeline
- Questionnaire-to-profile-vector conversion
- Multi-signal scoring formula
- Topic mismatch detection algorithm
- Rocchio feedback update derivation
- Explainability engine design
- BM25 hybrid scoring details
- Query expansion via TF-IDF + profile vectors
- Evaluation metric definitions

---

## 🗺️ Roadmap

### Short-Term
- [ ] Click-through implicit feedback (treat result clicks as positive signals)
- [ ] Temporal decay for feedback (recent feedback weighted higher)
- [ ] A/B comparison mode (side-by-side personalized vs. baseline)

### Medium-Term
- [ ] FAISS IVF/HNSW for scaling to 500k+ passages
- [ ] Query auto-suggest based on profile similarity
- [ ] Learning-to-rank (LambdaMART or neural ranker)

### Long-Term / Research
- [ ] Collaborative filtering across users
- [ ] Fine-tuned embeddings on MS MARCO relevance judgments
- [ ] Multi-modal retrieval (images, tables)

---

## 📖 References

1. Tri Nguyen et al., "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset," 2016
2. J.J. Rocchio, "Relevance Feedback in Information Retrieval," in SMART Retrieval System, 1971
3. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP 2019
4. J. Johnson, M. Douze, H. Jégou, "Billion-scale similarity search with GPUs," IEEE Trans. Big Data, 2019
5. C. Manning, P. Raghavan, H. Schütze, "Introduction to Information Retrieval," Cambridge University Press, 2008
6. S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," Foundations and Trends in IR, 2009

---

