import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import os

class IREngine:
    def __init__(self, model_name='all-MiniLM-L6-v2', subset_size=5000):
        self.model = SentenceTransformer(model_name)
        self.subset_size = subset_size
        self.documents = pd.DataFrame()
        self.embeddings = None
        self.index = None
        
        # Persistent storage: data/ folder with Parquet + FAISS index
        self.storage_dir = "data"
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            
        self.data_cache_path = os.path.join(self.storage_dir, f"msmarco_{subset_size}.parquet")
        self.index_cache_path = os.path.join(self.storage_dir, f"msmarco_{subset_size}.index")

    def load_data(self):
        """Loads MS MARCO data, using Parquet cache if available."""
        if os.path.exists(self.data_cache_path):
            print(f"✓ Loading cached data from: {self.data_cache_path}")
            self.documents = pd.read_parquet(self.data_cache_path)
            return True
        else:
            print(f"⬇ Downloading MS MARCO from HuggingFace (subset: {self.subset_size})...")
            dataset = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
            
            data = []
            for i, entry in enumerate(dataset):
                if i >= self.subset_size:
                    break
                for passage in entry['passages']['passage_text']:
                    data.append({
                        "id": len(data),
                        "content": passage,
                        "title": " ".join(passage.split()[:6]) + "..."
                    })
                    if len(data) >= self.subset_size:
                        break
                if len(data) >= self.subset_size:
                    break
            
            self.documents = pd.DataFrame(data)
            self.documents.to_parquet(self.data_cache_path, index=False)
            print(f"✓ Cached {len(self.documents)} passages to: {self.data_cache_path}")
            return False

    def build_index(self, batch_size=256):
        """Builds FAISS index, using local cache if available. Encodes in batches to save memory."""
        if os.path.exists(self.index_cache_path):
            print(f"✓ Loading FAISS index from: {self.index_cache_path}")
            self.index = faiss.read_index(self.index_cache_path)
            # Reconstruct embeddings for hybrid scoring
            self.embeddings = np.zeros((self.index.ntotal, self.index.d), dtype='float32')
            for i in range(self.index.ntotal):
                self.embeddings[i] = self.index.reconstruct(i)
            print(f"✓ Loaded {self.index.ntotal} vectors (dim={self.index.d})")
            return

        print("⚙ Encoding embeddings (first-time setup, this may take a minute)...")
        texts = self.documents['content'].tolist()
        
        # Encode in batches to reduce peak memory usage
        all_embeddings = []
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch = texts[start:end]
            batch_emb = self.model.encode(batch, show_progress_bar=False)
            all_embeddings.append(batch_emb)
            print(f"  Encoded {end}/{len(texts)} passages...")
        
        self.embeddings = np.vstack(all_embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings)
        
        # Save FAISS index to disk
        faiss.write_index(self.index, self.index_cache_path)
        print(f"✓ Index built and saved to: {self.index_cache_path}")

    def get_embedding(self, text):
        """Get a normalized embedding vector for a text string."""
        vec = self.model.encode([text])[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def search(self, query, profile_vec=None, personalization_weight=0.5):
        """Search with optional personalization via user profile vector."""
        q_vec = self.get_embedding(query)
        
        query_sims = np.dot(self.embeddings, q_vec)
        
        profile_sims = np.zeros(len(self.documents))
        if profile_vec is not None:
            profile_sims = np.dot(self.embeddings, profile_vec)
        
        q_weight = 1.0 - personalization_weight
        scores = (q_weight * query_sims) + (personalization_weight * profile_sims)
        
        results = self.documents.copy()
        results['score'] = scores
        results['query_sim'] = query_sims
        results['profile_sim'] = profile_sims
        
        return results.sort_values('score', ascending=False)

    def rocchio_update(self, current_profile, positive_vecs, negative_vecs, alpha=1.0, beta=0.75, gamma=0.25):
        """Rocchio relevance feedback to adapt user profile."""
        if current_profile is None:
            current_profile = np.zeros(self.index.d)
            
        pos_centroid = np.mean(positive_vecs, axis=0) if len(positive_vecs) > 0 else np.zeros_like(current_profile)
        neg_centroid = np.mean(negative_vecs, axis=0) if len(negative_vecs) > 0 else np.zeros_like(current_profile)
        
        new_profile = (alpha * current_profile) + (beta * pos_centroid) - (gamma * neg_centroid)
        
        norm = np.linalg.norm(new_profile)
        return new_profile / norm if norm > 0 else new_profile
