import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import os
import re
from collections import defaultdict

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
    def __init__(self, model_name='all-MiniLM-L6-v2', subset_size=5000):
        self.model = SentenceTransformer(model_name)
        self.subset_size = subset_size
        self.documents = pd.DataFrame()
        self.embeddings = None
        self.index = None

        # Persistent storage
        self.storage_dir = "data"
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        self.data_cache_path = os.path.join(self.storage_dir, f"msmarco_{subset_size}.parquet")
        self.index_cache_path = os.path.join(self.storage_dir, f"msmarco_{subset_size}.index")

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
            return

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
               preferred_sources=None, recency_weight=0.0, source_weight=0.0):
        """Search with multi-signal scoring: query sim + profile sim + source boost + recency."""
        q_vec = self.get_embedding(query)

        query_sims = np.dot(self.embeddings, q_vec)

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
        results['profile_sim'] = profile_sims
        results['source_boost'] = source_boosts
        results['recency_boost'] = recency_scores

        return results.sort_values('score', ascending=False)

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
