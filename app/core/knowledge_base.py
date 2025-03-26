import os
import time
import faiss
import pickle
import numpy as np


class KnowledgeBase:
    """Lightweight knowledge base for document storage and retrieval"""
    def __init__(self, cache_dir="kb_cache", embedding_dim=384):
        self.chunks = []
        self.chunk_metadata = []  # Store source file, position, etc.
        self.cache_dir = cache_dir
        self.embedding_dim = embedding_dim
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(embedding_dim)
        
        # For optimizing frequent queries
        self.query_cache = {}
        self.recent_queries = []
        self.max_cache_size = 100
        
    def add_document(self, file_path, doc_processor):
        """Process document and add to knowledge base"""
        start_time = time.time()
        
        try:
            text = doc_processor.process_document(file_path)
            if text.startswith("Error") or text.startswith("Unsupported"):
                return text
                
            # Create chunks with metadata
            chunks = doc_processor.create_chunks(text)
            source_name = os.path.basename(file_path)
            
            for i, chunk in enumerate(chunks):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    'source': source_name,
                    'index': i,
                    'type': os.path.splitext(file_path)[1]
                })
            
            # Create and store embeddings
            embeddings = doc_processor.create_embeddings(chunks)
            
            # Add to FAISS index
            self.index.add(embeddings)

            process_time = time.time() - start_time
            return f"Added {len(chunks)} chunks from {source_name} in {process_time:.2f}s"
    
    
        except RuntimeError as e:
            return str(e)
        
    def retrieve_relevant(self, query, doc_processor, top_k=3):
        """Retrieve most relevant chunks for a query"""
        if not self.chunks:
            return []
        
        # Check if query is in cache
        if query in self.query_cache:
            return self.query_cache[query]
        
        # Create query embedding
        query_embedding = doc_processor.embedding_model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search in FAISS index
        D, I = self.index.search(query_embedding, top_k)
        
        results = []
        for i, idx in enumerate(I[0]):
            if idx < len(self.chunks):  # Safety check
                results.append({
                    'text': self.chunks[idx],
                    'distance': D[0][i],  # L2 distance (lower is better)
                    'metadata': self.chunk_metadata[idx]
                })
        
        # Cache the results
        self._update_query_cache(query, results)
        
    def _update_query_cache(self, query, results):
        """Update query cache with LRU mechanism"""
        self.query_cache[query] = results
        self.recent_queries.append(query)
        
        # Remove oldest queries if cache is too large
        if len(self.recent_queries) > self.max_cache_size:
            oldest_query = self.recent_queries.pop(0)
            if oldest_query in self.query_cache:
                del self.query_cache[oldest_query]

    def retrieve_with_mmr(self, query, doc_processor, top_k=5, diversity=0.3):
        """Retrieve relevant chunks with Maximum Marginal Relevance for diversity"""
        if not self.chunks:
            return []
            
        # Get more candidates than needed for MMR
        candidates_k = min(top_k * 3, len(self.chunks))
        
        # Create query embedding
        query_embedding = doc_processor.embedding_model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search in FAISS index
        D, I = self.index.search(query_embedding, candidates_k)
        
        # Apply MMR to select diverse results
        selected_indices = []
        candidate_indices = [idx for idx in I[0] if idx < len(self.chunks)]
        
        # Get embeddings for all candidates
        candidate_embeddings = []
        for idx in candidate_indices:
            
            # We need to extract the embedding from the FAISS index
            # embedding = np.zeros((1, self.embedding_dim), dtype='float32')
            
            embedding = self.index.reconstruct(int(idx))
            candidate_embeddings.append(embedding)
        
        candidate_embeddings = np.array(candidate_embeddings)
        
        # Select first document with highest relevance
        selected_indices.append(0)  # First result is the most relevant
        
        # Select remaining documents
        while len(selected_indices) < top_k and len(candidate_indices) > len(selected_indices):
            best_score = -np.inf
            best_idx = -1
            
            for i in range(len(candidate_indices)):
                if i in selected_indices:
                    continue
                    
                # Calculate relevance (negative distance to query)
                relevance = -D[0][i]
                
                # Calculate diversity (max distance to already selected)
                diversity_values = []
                for j in selected_indices:
                    dist = np.linalg.norm(candidate_embeddings[i] - candidate_embeddings[j])
                    diversity_values.append(dist)
                
                diversity_score = min(diversity_values) if diversity_values else 0
                
                # Combine relevance and diversity
                score = (1 - diversity) * relevance + diversity * diversity_score
                
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx >= 0:
                selected_indices.append(best_idx)
            else:
                break
                
        # Create results
        results = []
        for i in selected_indices:
            idx = candidate_indices[i]
            results.append({
                'text': self.chunks[idx],
                'distance': D[0][i],  # L2 distance (lower is better)
                'metadata': self.chunk_metadata[idx]
            })
            
        return results
    
    def save(self):
        """Save knowledge base to disk"""
        with open(f"{self.cache_dir}/kb_data.pkl", 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'chunk_metadata': self.chunk_metadata
            }, f)
            
        # Save FAISS index
        faiss.write_index(self.index, f"{self.cache_dir}/kb_faiss.index")
    
    def load(self):
        """Load knowledge base from disk"""
        try:
            with open(f"{self.cache_dir}/kb_data.pkl", 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.chunk_metadata = data['chunk_metadata']
            
            # Load FAISS index
            self.index = faiss.read_index(f"{self.cache_dir}/kb_faiss.index")
            return f"Loaded {len(self.chunks)} chunks from cache"
        except:
            return "No cache found or error loading cache"
