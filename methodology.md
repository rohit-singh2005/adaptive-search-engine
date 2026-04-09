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

---

## 5. Future Scope

### 5.1 Short-Term

- **BM25 hybrid scoring**: Combine dense retrieval with sparse keyword matching (BM25/TF-IDF)
- **Query expansion**: Use profile vector to expand queries before retrieval
- **Evaluation metrics**: Compute precision, recall, nDCG, MAP against MS MARCO relevance labels
- **Configurable personalization slider**: Let users adjust the personalization weight `w_p` in real-time
- **Feedback persistence**: Save/load user profiles and preferences to disk (JSON/pickle) across sessions

### 5.2 Medium-Term

- **Click-through implicit feedback**: Treat result clicks as positive signals without explicit thumbs up
- **Larger dataset support**: FAISS IVF or HNSW indexes for approximate search on 50k–500k passages
- **Query suggestions**: Auto-suggest based on user profile similarity to a query log
- **A/B comparison mode**: Side-by-side personalized vs. baseline for the same query
- **Temporal decay**: Weight recent feedback more heavily than older feedback

### 5.3 Long-Term / Research

- **Learning-to-rank**: Replace linear blending with LambdaMART or a neural ranker
- **Collaborative filtering**: Cross-user recommendations (users who liked X also liked Y)
- **Fine-tuned embeddings**: Fine-tune the embedding model on MS MARCO relevance judgments
- **Multi-modal retrieval**: Support image/table retrieval alongside text
- **Real metadata**: Integrate actual timestamps and source URLs for genuine recency/source scoring

---

## 6. References

1. Tri Nguyen et al., "MS MARCO: A Human Generated MAchine Reading COmprehension Dataset," 2016
2. J.J. Rocchio, "Relevance Feedback in Information Retrieval," in SMART Retrieval System, 1971
3. N. Reimers and I. Gurevych, "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks," EMNLP 2019
4. J. Johnson, M. Douze, H. Jégou, "Billion-scale similarity search with GPUs," IEEE Trans. Big Data, 2019
5. C. Manning, P. Raghavan, H. Schütze, "Introduction to Information Retrieval," Cambridge University Press, 2008
