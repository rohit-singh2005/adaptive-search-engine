import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import re
import json
from collections import defaultdict
from datetime import datetime

# --- Topic keywords for classification ---
TOPIC_KEYWORDS = {
    "Tech": ["software", "computer", "programming", "algorithm", "code", "developer", "api",
             "machine learning", "artificial intelligence", "data", "technology", "app", "digital",
             "internet", "cyber", "hardware", "server", "cloud", "database", "network"],
    "Health": ["health", "medical", "doctor", "disease", "symptom", "treatment", "hospital",
               "nutrition", "fitness", "exercise", "mental health", "therapy", "vitamin",
               "diet", "wellness", "medicine", "clinical", "patient", "surgery", "cancer"],
    "Finance": ["finance", "stock", "market", "investment", "bank", "money", "economy",
                "trading", "cryptocurrency", "bitcoin", "budget", "loan", "mortgage",
                "tax", "revenue", "profit", "accounting", "insurance", "inflation", "debt"],
    "Sports": ["sports", "football", "basketball", "soccer", "tennis", "cricket", "athlete",
               "championship", "tournament", "league", "team", "game", "match", "score",
               "player", "coach", "olympic", "fitness", "stadium", "race"],
    "Entertainment": ["movie", "film", "music", "celebrity", "actor", "singer", "concert",
                      "series", "television", "show", "streaming", "album", "song", "dance",
                      "theater", "comedy", "drama", "anime", "gaming", "festival"],
    "Politics": ["politics", "government", "election", "president", "congress", "senate",
                 "democrat", "republican", "legislation", "policy", "campaign", "vote",
                 "political", "minister", "parliament", "diplomacy", "regulation"],
    "Gossip": ["gossip", "rumor", "scandal", "celebrity drama", "paparazzi", "tabloid",
               "controversy", "affair", "breakup", "dating", "relationship drama"],
    "Ads": ["advertisement", "sponsored", "promotion", "buy now", "discount", "deal",
            "limited offer", "subscribe", "click here", "free trial", "shop now"]
}

# --- Source-type keywords ---
SOURCE_KEYWORDS = {
    "official": ["government", "official", "department", ".gov", "regulation", "authority",
                 "institute", "university", "research", "journal", "published", "peer-reviewed"],
    "blog": ["blog", "opinion", "personal", "my experience", "i think", "perspective",
             "thoughts on", "review", "story", "journey"],
    "news": ["reported", "breaking", "according to", "sources say", "announcement",
             "press release", "headline", "coverage", "journalist", "reuters", "associated press"],
    "forum": ["forum", "question", "answer", "discussion", "thread", "comment", "reply",
              "posted by", "user said", "reddit", "quora", "stack overflow"]
}


class IREngine:
    def __init__(self, model_name='all-MiniLM-L6-v2', subset_size=50000):
        self.model = SentenceTransformer(model_name)
        self.subset_size = subset_size
        self.documents = pd.DataFrame()
        self.embeddings = None
        self.index = None
        self.bm25 = None          # BM25 sparse index
        self.tfidf = None          # TF-IDF vectorizer for query expansion

        # Persistent storage
        self.storage_dir = "data"
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        self.data_cache_path = os.path.join(self.storage_dir, f"msmarco_{subset_size}.parquet")
        self.index_cache_path = os.path.join(self.storage_dir, f"msmarco_{subset_size}.index")
        self.profiles_path = os.path.join(self.storage_dir, "user_profiles.json")

    def load_data(self):
        """Loads MS MARCO data, using Parquet cache if available."""
        if os.path.exists(self.data_cache_path):
            print(f"Loading cached data from: {self.data_cache_path}")
            self.documents = pd.read_parquet(self.data_cache_path)
            # Ensure source_type column exists (for older caches)
            if 'source_type' not in self.documents.columns:
                self.documents['source_type'] = self.documents['content'].apply(self._detect_source_type)
                self.documents.to_parquet(self.data_cache_path, index=False)
            return True
        else:
            print(f"Downloading MS MARCO from HuggingFace (subset: {self.subset_size})...")
            dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)

            data = []
            for i, entry in enumerate(dataset):
                if i >= self.subset_size:
                    break
                for passage in entry['passages']['passage_text']:
                    source_type = self._detect_source_type(passage)
                    data.append({
                        "id": len(data),
                        "content": passage,
                        "title": " ".join(passage.split()[:6]) + "...",
                        "source_type": source_type,
                    })
                    if len(data) >= self.subset_size:
                        break
                if len(data) >= self.subset_size:
                    break

            self.documents = pd.DataFrame(data)
            self.documents.to_parquet(self.data_cache_path, index=False)
            print(f"Cached {len(self.documents)} passages to: {self.data_cache_path}")
            return False

    def build_index(self, batch_size=256):
        """Builds FAISS index, using local cache if available."""
        if os.path.exists(self.index_cache_path):
            print(f"Loading FAISS index from: {self.index_cache_path}")
            self.index = faiss.read_index(self.index_cache_path)
            self.embeddings = np.zeros((self.index.ntotal, self.index.d), dtype='float32')
            for i in range(self.index.ntotal):
                self.embeddings[i] = self.index.reconstruct(i)
            print(f"Loaded {self.index.ntotal} vectors (dim={self.index.d})")
        else:
            print("Encoding embeddings (first-time setup)...")
            texts = self.documents['content'].tolist()

            all_embeddings = []
            for start in range(0, len(texts), batch_size):
                end = min(start + batch_size, len(texts))
                batch = texts[start:end]
                batch_emb = self.model.encode(batch, show_progress_bar=False)
                all_embeddings.append(batch_emb)
                print(f"  Encoded {end}/{len(texts)} passages...")

            self.embeddings = np.vstack(all_embeddings).astype('float32')
            faiss.normalize_L2(self.embeddings)

            dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(self.embeddings)

            faiss.write_index(self.index, self.index_cache_path)
            print(f"Index built and saved to: {self.index_cache_path}")

        # Always build BM25 + TF-IDF (fast, in-memory)
        self._build_bm25_index()
        self._build_tfidf()

    # ------------------------------------------------------------------
    #  BM25 Sparse Index
    # ------------------------------------------------------------------

    def _build_bm25_index(self):
        """Build BM25 index from document contents for sparse keyword matching."""
        print("Building BM25 index...")
        texts = self.documents['content'].tolist()
        tokenized = [self._tokenize(doc) for doc in texts]
        self.bm25 = BM25Okapi(tokenized)
        print(f"BM25 index built over {len(texts)} documents")

    def _build_tfidf(self):
        """Build TF-IDF vectorizer for query expansion keyword extraction."""
        print("Fitting TF-IDF vectorizer...")
        self.tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
        self.tfidf.fit(self.documents['content'].tolist())
        print("TF-IDF vectorizer ready")

    @staticmethod
    def _tokenize(text):
        """Simple whitespace + lowercase tokenizer for BM25."""
        return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower()).split()

    def bm25_scores(self, query):
        """Get BM25 scores for all documents given a query."""
        if self.bm25 is None:
            return np.zeros(len(self.documents))
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        # Normalize to [0, 1]
        max_s = scores.max()
        return scores / max_s if max_s > 0 else scores

    # ------------------------------------------------------------------
    #  Query Expansion
    # ------------------------------------------------------------------

    def expand_query(self, query, profile_vec, n_terms=5, n_neighbors=20):
        """Expand query with terms from profile-similar documents using TF-IDF.
        
        Finds documents similar to the user's profile, extracts their top TF-IDF
        terms, and appends the most relevant ones to the original query.
        """
        if profile_vec is None or np.linalg.norm(profile_vec) == 0:
            return query
        if self.tfidf is None:
            return query

        # Find documents most similar to the user's profile
        profile_sims = np.dot(self.embeddings, profile_vec)
        top_indices = np.argsort(profile_sims)[-n_neighbors:]
        profile_docs = [self.documents.iloc[i]['content'] for i in top_indices]

        # Extract top TF-IDF terms from those documents
        tfidf_matrix = self.tfidf.transform(profile_docs)
        feature_names = np.array(self.tfidf.get_feature_names_out())

        # Average TF-IDF scores across profile-similar documents
        avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()

        # Filter out terms already in the query
        query_words = set(query.lower().split())
        candidates = []
        for idx in np.argsort(avg_scores)[::-1]:
            term = feature_names[idx]
            if term not in query_words and len(term) > 2:
                candidates.append(term)
            if len(candidates) >= n_terms:
                break

        expanded = query + " " + " ".join(candidates)
        return expanded

    def get_embedding(self, text):
        """Get a normalized embedding vector for a text string."""
        vec = self.model.encode([text])[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    # ------------------------------------------------------------------
    #  Topic classification & source detection
    # ------------------------------------------------------------------

    @staticmethod
    def classify_text(text, keyword_map=TOPIC_KEYWORDS):
        """Classify text into topics based on keyword matching. Returns list of matched topics."""
        text_lower = text.lower()
        matches = {}
        for topic, keywords in keyword_map.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > 0:
                matches[topic] = count
        if matches:
            sorted_topics = sorted(matches.items(), key=lambda x: x[1], reverse=True)
            return [t[0] for t in sorted_topics]
        return []

    @staticmethod
    def _detect_source_type(text):
        """Detect the likely source type of a passage."""
        text_lower = text.lower()
        scores = {}
        for src, keywords in SOURCE_KEYWORDS.items():
            scores[src] = sum(1 for kw in keywords if kw in text_lower)
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "general"

    def detect_topic_conflict(self, query, avoid_topics):
        """Check if query topic conflicts with user's avoided topics.
        Returns (has_conflict, adjusted_personalization_weight).
        """
        query_topics = self.classify_text(query)
        for topic in query_topics:
            if topic in avoid_topics:
                return True, 0.2  # reduce personalization weight
        return False, 0.5  # normal weight

    # ------------------------------------------------------------------
    #  Questionnaire → initial profile
    # ------------------------------------------------------------------

    def build_initial_profile(self, interests, depth, source_pref):
        """Convert questionnaire answers into an initial profile vector."""
        parts = []

        # Interests contribute the most
        if interests:
            interest_text = " ".join(interests)
            parts.append(interest_text)

        # Depth preference adds bias toward certain content types
        depth_map = {
            "quick": "summary overview brief short",
            "detailed": "detailed comprehensive in-depth analysis explanation",
            "research": "research study paper findings methodology experiment"
        }
        if depth in depth_map:
            parts.append(depth_map[depth])

        # Source preference adds bias
        source_map = {
            "official": "official government academic peer-reviewed journal",
            "blog": "blog personal opinion experience review",
            "news": "news breaking report headline coverage",
            "forum": "forum discussion question answer community"
        }
        if source_pref in source_map:
            parts.append(source_map[source_pref])

        if not parts:
            return np.zeros(self.index.d)

        combined = " ".join(parts)
        return self.get_embedding(combined)

    # ------------------------------------------------------------------
    #  Search with multi-signal scoring
    # ------------------------------------------------------------------

    def search(self, query, profile_vec=None, personalization_weight=0.5,
               preferred_sources=None, recency_weight=0.0, source_weight=0.0,
               dense_weight=0.7, use_bm25=True, expand=False):
        """Search with multi-signal scoring: dense + BM25 + profile + source + recency.
        
        Args:
            query: Search query string
            profile_vec: User profile embedding vector
            personalization_weight: Weight for profile similarity (0-1)
            preferred_sources: List of preferred source types for boosting
            recency_weight: Weight for recency boost (0-0.15)
            source_weight: Weight for source type boost
            dense_weight: Blend between dense (1.0) and BM25 (0.0) for base relevance
            use_bm25: Whether to include BM25 in scoring
            expand: Whether to apply query expansion using profile
        """
        # Optionally expand query with profile-derived terms
        effective_query = query
        expansion_terms = []
        if expand and profile_vec is not None:
            effective_query = self.expand_query(query, profile_vec)
            expansion_terms = effective_query.replace(query, "").strip().split()

        q_vec = self.get_embedding(effective_query)

        # Dense similarity
        dense_sims = np.dot(self.embeddings, q_vec)

        # BM25 sparse similarity
        bm25_sims = np.zeros(len(self.documents))
        if use_bm25 and self.bm25 is not None:
            bm25_sims = self.bm25_scores(effective_query)

        # Hybrid base relevance: blend dense + BM25
        query_sims = dense_weight * dense_sims + (1.0 - dense_weight) * bm25_sims

        # Profile similarity
        profile_sims = np.zeros(len(self.documents))
        if profile_vec is not None and np.linalg.norm(profile_vec) > 0:
            profile_sims = np.dot(self.embeddings, profile_vec)

        # Source boost: boost documents matching preferred source types
        source_boosts = np.zeros(len(self.documents))
        if preferred_sources and 'source_type' in self.documents.columns:
            source_boosts = self.documents['source_type'].apply(
                lambda s: 1.0 if s in preferred_sources else 0.0
            ).values.astype('float32')

        # Recency boost: simulate via document position (lower id = earlier in dataset)
        # In a real system this would use actual timestamps
        n = len(self.documents)
        recency_scores = np.linspace(0.0, 1.0, n).astype('float32')  # higher id = more "recent"

        # Weighted combination
        q_w = 1.0 - personalization_weight
        scores = (
            q_w * query_sims
            + personalization_weight * profile_sims
            + source_weight * source_boosts
            + recency_weight * recency_scores
        )

        results = self.documents.copy()
        results['score'] = scores
        results['query_sim'] = query_sims
        results['dense_sim'] = dense_sims
        results['bm25_sim'] = bm25_sims
        results['profile_sim'] = profile_sims
        results['source_boost'] = source_boosts
        results['recency_boost'] = recency_scores

        return results.sort_values('score', ascending=False), expansion_terms

    # ------------------------------------------------------------------
    #  Explainability
    # ------------------------------------------------------------------

    def explain_result(self, result_row, query, liked_docs_text):
        """Generate a simple explanation of why a result was ranked highly."""
        reasons = []

        # Query relevance
        q_sim = result_row.get('query_sim', 0)
        if q_sim > 0.5:
            reasons.append(f"Strong match to your query ({int(q_sim*100)}% similarity)")
        elif q_sim > 0.3:
            reasons.append(f"Moderate match to your query ({int(q_sim*100)}% similarity)")

        # Profile relevance
        p_sim = result_row.get('profile_sim', 0)
        if p_sim > 0.3:
            reasons.append("Aligns with your learned interests")

        # Source boost
        s_boost = result_row.get('source_boost', 0)
        if s_boost > 0:
            reasons.append(f"Matches your preferred source type: {result_row.get('source_type', 'N/A')}")

        # Similarity to liked docs
        if liked_docs_text:
            content = result_row['content'].lower()
            for liked_text in liked_docs_text[-3:]:  # check last 3 liked
                # Simple word overlap check
                liked_words = set(liked_text.lower().split())
                content_words = set(content.split())
                overlap = liked_words & content_words
                # Filter out very common words
                overlap -= {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
                            "and", "for", "on", "it", "that", "this", "with", "as", "by", "at"}
                if len(overlap) > 5:
                    sample = list(overlap)[:3]
                    reasons.append(f"Similar to a document you liked (shared terms: {', '.join(sample)})")
                    break

        if not reasons:
            reasons.append("Baseline relevance to your query")

        return reasons

    # ------------------------------------------------------------------
    #  Rocchio feedback
    # ------------------------------------------------------------------

    def rocchio_update(self, current_profile, positive_vecs, negative_vecs, alpha=1.0, beta=0.75, gamma=0.25):
        """Rocchio relevance feedback to adapt user profile."""
        if current_profile is None:
            current_profile = np.zeros(self.index.d)

        pos_centroid = np.mean(positive_vecs, axis=0) if len(positive_vecs) > 0 else np.zeros_like(current_profile)
        neg_centroid = np.mean(negative_vecs, axis=0) if len(negative_vecs) > 0 else np.zeros_like(current_profile)

        new_profile = (alpha * current_profile) + (beta * pos_centroid) - (gamma * neg_centroid)

        norm = np.linalg.norm(new_profile)
        return new_profile / norm if norm > 0 else new_profile

    # ------------------------------------------------------------------
    #  Profile Persistence
    # ------------------------------------------------------------------

    def save_profiles(self, user_profiles, user_prefs, liked_docs, feedback_log=None):
        """Save user profiles, preferences, liked docs, and feedback log to disk as JSON."""
        data = {
            "profiles": {name: vec.tolist() for name, vec in user_profiles.items()},
            "prefs": user_prefs,
            "liked_docs": liked_docs,
            "feedback_log": feedback_log or {},
            "saved_at": datetime.now().isoformat(),
        }
        with open(self.profiles_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_profiles(self):
        """Load user profiles from disk. Returns (profiles_dict, prefs_dict, liked_docs, feedback_log) or None."""
        if not os.path.exists(self.profiles_path):
            return None
        try:
            with open(self.profiles_path, 'r') as f:
                data = json.load(f)
            profiles = {name: np.array(vec, dtype='float32') for name, vec in data.get("profiles", {}).items()}
            prefs = data.get("prefs", {})
            liked_docs = data.get("liked_docs", {})
            feedback_log = data.get("feedback_log", {})
            return profiles, prefs, liked_docs, feedback_log
        except (json.JSONDecodeError, KeyError):
            return None

    # ------------------------------------------------------------------
    #  Evaluation Metrics
    # ------------------------------------------------------------------

    def compute_precision_at_k(self, ranked_doc_ids, relevant_ids, k=10):
        """Compute Precision@K: fraction of top-K results that are relevant.
        
        Uses liked document IDs as the relevance set.
        """
        if not relevant_ids or k == 0:
            return 0.0
        top_k = ranked_doc_ids[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return relevant_in_top_k / k

    def compute_mrr(self, ranked_doc_ids, relevant_ids):
        """Compute Mean Reciprocal Rank: 1/rank of first relevant result.
        
        Uses liked document IDs as the relevance set.
        """
        if not relevant_ids:
            return 0.0
        for rank, doc_id in enumerate(ranked_doc_ids, 1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    def compute_recall_at_k(self, ranked_doc_ids, relevant_ids, k=10):
        """Compute Recall@K: fraction of relevant docs found in top-K."""
        if not relevant_ids or k == 0:
            return 0.0
        top_k = ranked_doc_ids[:k]
        found = sum(1 for doc_id in top_k if doc_id in relevant_ids)
        return found / len(relevant_ids)

    def evaluate(self, query, profile_vec, relevant_ids, **search_kwargs):
        """Run a full evaluation: search + compute P@5, P@10, MRR, Recall@10.
        
        Returns dict of metrics.
        """
        results, _ = self.search(query, profile_vec=profile_vec, **search_kwargs)
        ranked_ids = results['id'].tolist()
        rel_set = set(relevant_ids)

        return {
            "P@5": self.compute_precision_at_k(ranked_ids, rel_set, k=5),
            "P@10": self.compute_precision_at_k(ranked_ids, rel_set, k=10),
            "MRR": self.compute_mrr(ranked_ids, rel_set),
            "Recall@10": self.compute_recall_at_k(ranked_ids, rel_set, k=10),
        }
