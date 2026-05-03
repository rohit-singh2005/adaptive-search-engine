# Adaptive Search Engine — Methodology

## 1. Problem Statement

Traditional search engines return the same results for every user regardless of their individual preferences and past behavior. This project implements a **retrieval system that adapts search results based on user behavior and interests**, using initial preference capture, multi-signal scoring, implicit/explicit relevance feedback, and explainable ranking to personalize results in real-time.

---

## 2. System Architecture

```
┌──────────────────┐     ┌─────────────────────────┐     ┌────────────────┐
│  Streamlit UI    │────▶│     IREngine Core        │────▶│  FAISS Index   │
│  (app.py)        │◀────│   (ir_engine.py)         │◀────│ (IndexFlatIP)  │
│                  │     │                          │     └────────────────┘
│ • User Selection │     │ • SentenceTransformer    │
│ • Questionnaire  │     │ • Topic Classification   │     ┌────────────────┐
│ • Search         │     │ • Source Detection        │────▶│  Disk Cache    │
│ • Feedback       │     │ • Mismatch Detection      │     │  (Parquet +    │
│ • Explainability │     │ • Multi-Signal Scoring    │     │   FAISS .index)│
│ • Conflict Warn  │     │ • Rocchio Feedback        │     └────────────────┘
│                  │     │ • Explainability Engine    │
└──────────────────┘     └─────────────────────────┘
```

### Component Summary

| Component | Technology | Purpose |
|---|---|---|
| Embedding Model | `all-MiniLM-L6-v2` (384-dim) | Dense vector encoding for passages and queries |
| Vector Index | FAISS `IndexFlatIP` | Exact cosine similarity search |
| Data Store | Parquet + FAISS `.index` | Disk caching for fast restarts |
| User Profiling | Rocchio Algorithm | Updates interest vectors from relevance feedback |
| Topic Classifier | Keyword-based | Classifies queries/passages into predefined topics |
| Source Detector | Keyword-based | Infers passage source type (official, blog, news, forum) |
| Explainability | Multi-factor reasoning | Generates human-readable explanations for rankings |
| Frontend | Streamlit | Interactive UI with questionnaire, search, and feedback |

---

## 3. Methodology

### 3.1 Data Ingestion

- **Source**: MS MARCO v1.1 passage dataset (HuggingFace, streaming mode)
- **Subset**: 5,000 passages (configurable) to keep memory/compute manageable
- **Processing**: Each passage is enriched with:
  - `title`: Auto-generated from first 6 words
  - `source_type`: Detected via keyword matching (official / blog / news / forum / general)
- **Caching**: First run saves to Parquet; subsequent runs load from disk instantly

### 3.2 Embedding and Indexing

1. Passages encoded in mini-batches of 256 using `all-MiniLM-L6-v2`
2. All vectors L2-normalized so inner product = cosine similarity
3. Indexed in FAISS `IndexFlatIP` (exact search)
4. Index saved to disk for instant loading on restart

### 3.3 Initial Questionnaire Module

Before the first search, each user completes a preference questionnaire:

| Preference | Options | Effect |
|---|---|---|
| Interests | Tech, Health, Finance, Sports, Entertainment | Encoded into initial profile vector |
| Search Depth | quick, detailed, research | Biases profile toward summary vs. in-depth content |
| Recency | last week / month / year / any | Controls recency weight in scoring formula |
| Source Type | official, blog, news, forum | Boosts results from preferred source type |
| Avoid Topics | Politics, Gossip, Ads | Triggers mismatch detection to reduce personalization |

**Profile initialization:**
```
profile_vec = encode("interests + depth_keywords + source_keywords")
```

The questionnaire answers are converted to a natural language string and encoded into a dense vector, giving the user a non-zero starting profile before any feedback.

### 3.4 Multi-Signal Scoring

The ranking formula combines four signals:

```
final_score = w_q × query_sim + w_p × profile_sim + w_s × source_boost + w_r × recency_boost
```

| Signal | Weight | Description |
|---|---|---|
| `query_sim` | `1 - w_p` | Cosine similarity between query and passage embeddings |
| `profile_sim` | `0.5` (default) | Cosine similarity between user profile and passage embeddings |
| `source_boost` | `0.1` | Binary boost if passage source type matches user preference |
| `recency_boost` | `0.0 – 0.15` | Based on recency preference (higher = prefers recent content) |

### 3.5 Query–Topic Mismatch Detection

Before executing a search, the system classifies the query into topics using keyword matching:

```python
def detect_topic_conflict(query, user_avoid_topics):
    query_topics = classify_query(query)
    if any(topic in user_avoid_topics for topic in query_topics):
        return True, 0.2  # reduce personalization weight to 20%
    return False, 0.5     # normal weight
```

When a conflict is detected:
- A warning banner is displayed to the user
- Personalization weight is reduced from 0.5 → 0.2
- Results lean more toward baseline relevance

### 3.6 Personalization via Rocchio Relevance Feedback

Users provide explicit feedback (thumbs up / thumbs down) on individual results.

**Rocchio update formula:**
```
profile_new = α × profile_current + β × centroid(positive) − γ × centroid(negative)
```

| Parameter | Value | Role |
|---|---|---|
| α | 1.0 | Retention of existing profile |
| β | 0.75 | Weight of positive feedback |
| γ | 0.25 | Weight of negative feedback |

The updated profile is normalized to unit length. Each user maintains an independent profile vector.

### 3.7 Explainable Feedback Panel

Each personalized result includes a human-readable explanation of its ranking. The explanation engine checks:

1. **Query similarity** — how well the passage matches the query text
2. **Profile alignment** — whether the passage aligns with learned user interests
3. **Source match** — whether the passage comes from the user's preferred source type
4. **Liked-document similarity** — word overlap with previously liked documents

Example output:
```
· Strong match to your query (72% similarity)
· Aligns with your learned interests
· Similar to a document you liked (shared terms: python, machine, learning)
```

### 3.8 Multi-User Support

- 5 independent user profiles (User 1 – User 5)
- Each user has: separate profile vector, questionnaire preferences, feedback history
- Switching users loads a completely different personalization context

---

## 4. Current Implementation Status

| Feature | Status |
|---|---|
| MS MARCO streaming + Parquet caching | Done |
| FAISS index with disk caching | Done |
| Batched SentenceTransformer encoding | Done |
| Initial questionnaire (interests/depth/recency/source/avoid) | Done |
| Questionnaire → initial profile vector | Done |
| Multi-signal scoring (query + profile + source + recency) | Done |
| Query–topic mismatch detection + warning | Done |
| Rocchio relevance feedback (thumbs up/down) | Done |
| Explainable feedback panel per result | Done |
| Source type detection + badges | Done |
| Multi-user profiles (5 users) | Done |
| Row-based result layout with Show More pagination | Done |
| **BM25 Hybrid Scoring (sparse + dense)** | **Done** |
| **Query Expansion via profile vectors + TF-IDF** | **Done** |
| **Profile Persistence (JSON on disk)** | **Done** |
| **Personalization Weight Slider (real-time control)** | **Done** |
| **Search Analytics Dashboard (topic affinity chart, profile strength)** | **Done** |
| **Query History & Session Logging** | **Done** |
| **Evaluation Metrics (P@5, P@10, MRR, Recall@10)** | **Done** |

---

## 5. New Features (Detailed)

### 5.1 BM25 Hybrid Scoring

The system now combines **dense retrieval** (SentenceTransformer embeddings + FAISS cosine similarity) with **sparse retrieval** (BM25 Okapi term-frequency scoring). The hybrid formula is:

```
base_relevance = α × dense_sim + (1 − α) × bm25_sim
```

Where `α` (dense weight) is user-controllable via a sidebar slider (default 0.7). This captures both semantic similarity and exact keyword matching, which is the gold standard in modern IR systems.

- BM25 index is built in-memory on every startup (fast, ~1 second for 5K docs)
- Scores are normalized to [0, 1] before blending
- Each result row shows both Dense% and BM25% breakdown badges

### 5.2 Query Expansion

When enabled, the system **automatically expands queries** with terms derived from the user's interest profile:

1. Find top-N documents most similar to the user's profile vector
2. Extract their highest-scoring TF-IDF terms
3. Filter out terms already present in the query
4. Append top-5 expansion terms to the query before retrieval

This improves recall for users with established profiles. Expansion terms are displayed in a banner above results.

### 5.3 Profile Persistence

User profiles, preferences, liked documents, and feedback logs are now **saved to disk as JSON** (`data/user_profiles.json`):

- **Auto-save**: Triggers after every questionnaire submission and every feedback action
- **Auto-load**: On app startup, previously saved profiles are restored
- **Format**: JSON with numpy arrays serialized as lists

### 5.4 Real-Time Personalization Controls

A sidebar panel provides three controls:

| Control | Range | Default | Effect |
|---|---|---|---|
| Personalization Weight | 0.0 – 1.0 | 0.5 | How much profile similarity affects ranking |
| Dense vs BM25 Blend | 0.0 – 1.0 | 0.7 | Balance between semantic and keyword scoring |
| Query Expansion | On/Off | Off | Toggles profile-based query expansion |

### 5.5 Search Analytics Dashboard

The sidebar displays a live analytics panel:

- **Profile Strength Indicator**: Progress bar showing profile vector magnitude (0–100%)
- **Topic Affinity Chart**: Horizontal bar chart (Plotly) showing cosine similarity between the user's profile and each topic embedding
- **Feedback Counters**: Total likes and dislikes as metric cards
- **Recent Searches**: Last 5 queries displayed with timestamps

### 5.6 Query History & Session Logging

Every search is logged with:

```json
{"query": "...", "time": "ISO timestamp", "topics": ["Tech", "Health"]}
```

Every feedback action is logged with:

```json
{"action": "like/dislike", "doc_id": 42, "time": "ISO timestamp"}
```

These logs persist across sessions via the profile persistence system.

### 5.7 Evaluation Metrics

After providing feedback, an expandable "Retrieval Evaluation Metrics" panel appears showing:

| Metric | Formula | Description |
|---|---|---|
| P@5 | relevant_in_top5 / 5 | Precision at rank 5 |
| P@10 | relevant_in_top10 / 10 | Precision at rank 10 |
| MRR | 1 / rank_of_first_relevant | Mean Reciprocal Rank |
| Recall@10 | found_in_top10 / total_relevant | Recall at rank 10 |

Liked documents serve as the relevance set. A Plotly line chart tracks how P@5 and MRR evolve across successive search rounds, demonstrating the adaptive improvement over time.

---

## 6. Updated Architecture

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

---

## 7. Future Scope

### 7.1 Short-Term

- **Click-through implicit feedback**: Treat result clicks as positive signals
- **Temporal decay**: Weight recent feedback more heavily than older feedback
- **A/B comparison mode**: Side-by-side personalized vs. baseline for the same query

### 7.2 Medium-Term

- **Larger dataset support**: FAISS IVF or HNSW indexes for 50k–500k passages
- **Query suggestions**: Auto-suggest based on profile similarity
- **Learning-to-rank**: Replace linear blending with LambdaMART or neural ranker

### 7.3 Long-Term / Research

- **Collaborative filtering**: Cross-user recommendations
- **Fine-tuned embeddings**: Fine-tune on MS MARCO relevance judgments
- **Multi-modal retrieval**: Support image/table retrieval alongside text

---

## 8. References

1. Tri Nguyen et al., "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset," 2016
2. J.J. Rocchio, "Relevance Feedback in Information Retrieval," in SMART Retrieval System, 1971
3. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP 2019
4. J. Johnson, M. Douze, H. Jégou, "Billion-scale similarity search with GPUs," IEEE Trans. Big Data, 2019
5. C. Manning, P. Raghavan, H. Schütze, "Introduction to Information Retrieval," Cambridge University Press, 2008
6. S. Robertson and H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," Foundations and Trends in IR, 2009
