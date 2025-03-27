# app/chatbot/nl2sql.py
import os
import re
import time
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional

# from transformers import AutoTokenizer, AutoModel, pipeline # local call llm
from core.llm import SiliconFlowLLM # api call llm
from core.document_processor import DocumentProcessor 
from core.knowledge_base import KnowledgeBase
from core.database import DataBase

class NL2SQLChatbot:
    def __init__(self, documents_dir: str = "../data", embedding_dim: int = 384):
        """
        Initialize the NL2SQL Chatbot with document processing and database capabilities
        
        Args:
            documents_dir: Directory containing knowledge documents
            embedding_dim: Dimension of embedding vectors
        """
        # # Initialize the LLM
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name)
        # self.nlp_pipeline = pipeline("text2text-generation", model=model_name, max_length=1024)
        self.model_engine = SiliconFlowLLM()
        
        # Database connection
        self.db = DataBase()

        # Load schema info
        self.schema_info = self.db.get_schema_as_dict()
        
        # Document processing
        self.doc_processor = DocumentProcessor(embedding_dim=embedding_dim)
        self.knowledge_base = KnowledgeBase(embedding_dim=embedding_dim)
        
        # Try to load cached knowledge base
        start_time = time.time()
        load_result = self.knowledge_base.load()
        print("knowledge_base load_result: ", load_result)
        
        if "No cache" in load_result:
            # Process documents if no cache
            self._process_documents(documents_dir)
            # Save knowledge base for future use
            self.knowledge_base.save()
        else:
            print(f"Loaded knowledge base in {time.time() - start_time:.2f}s")
        
        # Conversation history
        self.conversation_history = []

        # Query optimization
        self.common_terms = Counter()
        
    def _process_documents(self, documents_dir):
        """Process all documents in the specified directory"""
        if not os.path.exists(documents_dir):
            print(f"Documents directory not found: {documents_dir}")
            return
            
        for filename in os.listdir(documents_dir):
            file_path = os.path.join(documents_dir, filename)
            if os.path.isfile(file_path):
                print(f"Processing {filename}...")
                result = self.knowledge_base.add_document(file_path, self.doc_processor)
                print(result)

    def _update_query_statistics(self, query):
        """Update statistics about common query terms for optimization"""
        # Simple tokenization
        terms = re.findall(r'\b\w+\b', query.lower())
        self.common_terms.update(terms)
    
    def _retrieve_relevant_knowledge(self, query):
        """Retrieve relevant information from knowledge base"""
        # Track query patterns to improve future retrievals
        self._update_query_statistics(query)

        # Use MMR for diverse results
        results = self.knowledge_base.retrieve_with_mmr(query, self.doc_processor, top_k=3, diversity=0.3)
        
        # results = self.knowledge_base.retrieve_relevant(query, self.doc_processor, top_k=3)
        
        if not results:
            return "No relevant information found."
            
        # Group results by source to improve context coherence
        source_groups = {}
        for result in results:
            source = result['metadata']['source']
            if source not in source_groups:
                source_groups[source] = []
            source_groups[source].append(result)
            
        relevant_text = ""
        
        # Add results, grouped by source
        for source, items in source_groups.items():
            relevant_text += f"[Document: {source}]\n"
            
            # Sort items by their original index to maintain document flow
            items.sort(key=lambda x: x['metadata']['index'])
            
            for item in items:
                relevant_text += f"{item['text']}\n\n"
            
        return relevant_text
    
    def _prepare_prompt(self, user_query):
        """Prepare a prompt for the language model"""
        # Retrieve relevant knowledge
        start_time = time.time()
        relevant_knowledge = self._retrieve_relevant_knowledge(user_query)
        retrieval_time = time.time() - start_time
        print(f"Knowledge retrieval time: {retrieval_time:.2f}s")
        
        # Format conversation history
        history_text = ""
        history_queries = []

        for turn in self.conversation_history[-3:]:  # Last 3 turns for context
            user_text = turn.get("user", "Unknown User Input")
            system_text = turn.get("system", "No System Response")
            history_text += f"User: {user_text}\nSystem: {system_text}\n"

        # Detect if this is a follow-up question
        is_followup = self._detect_followup_question(user_query, history_queries)
        

        # print(f"""=====chat_history=====\n{self.conversation_history}\n""")  # Debugging line
        # Format the prompt
        prompt = f"""
基于数据table schema：{self.schema_info}

和知识库相关内容：{relevant_knowledge}

以及历史会话记录：{history_text}
{'这是基于上一轮对话的一个follow-up。' if is_followup else ''}

生成合法的SQL SELECT语句，要求：
1. 根据已知信息判断，是否需要引导用户提问或者提问用户获取信息
2. 如果有足够信息，生成合法的SQL SELECT语句
3. 返回查询结果

如果模糊判断需要更多信息, respond with: "请提供具体信息: [specific question]"
如果用户需要引导，respond with:"引导提问: [specific instruction]"
如果已经有足够信息来生成sql, respond with: "SQL_QUERY: [SQL_QUERY]"
"""
        # print(f"""=====prompt=====\n{prompt}\n""")
        return prompt
    
    def _detect_followup_question(self, query, history_queries):
        """Detect if the current question is a follow-up to previous ones"""
        # Simple heuristics for follow-up detection
        
        # Check for pronouns that might refer to previous entities
        followup_indicators = ['it', 'they', 'them', 'those', 'that', 'these', 'this']
        query_lower = query.lower()
        
        for indicator in followup_indicators:
            if re.search(r'\b' + indicator + r'\b', query_lower):
                return True
                
        # Very short questions are often follow-ups
        if len(query.split()) <= 4:
            return True
            
        # No explicit table mention but previous queries had them
        table_names = list(self.schema_info.keys())
        has_table_reference = any(table.lower() in query_lower for table in table_names)
        
        if not has_table_reference and history_queries:
            # Check if previous queries mentioned tables
            prev_had_tables = any(
                any(table.lower() in prev.lower() for table in table_names)
                for prev in history_queries
            )
            if prev_had_tables:
                return True
                
        return False
    
    def _needs_clarification(self, query, generated_text):
        """Determine if we need more information from the user"""
        if "请提供具体信息:" in generated_text:
            return True, generated_text.split("请提供具体信息:")[1].strip()
        elif "引导提问:" in generated_text:
            return True, generated_text.split("引导提问:")[1].strip()
        return False, ""
    
    def _extract_sql_query(self, generated_text):
        """Extract SQL_QUERY from generated text"""
        if "SQL_QUERY:" in generated_text:
            raw_sql = generated_text.split("SQL_QUERY:")[1].strip()
            if "```sql" in raw_sql:
                clean_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
                return clean_sql
            return raw_sql
        return None
    
    def _validate_sql(self, query):
        """Enhanced SQL validation"""
        # Check for dangerous keywords
        dangerous_keywords = ['DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'UPDATE', 'INSERT', 'GRANT', 'REVOKE']
        for keyword in dangerous_keywords:
            if re.search(r'\b' + keyword + r'\b', query.upper()):
                return False, f"SQL contains potentially dangerous keyword: {keyword}"
        
        # Check for valid table references
        table_pattern = re.compile(r'FROM\s+(\w+)', re.IGNORECASE)
        join_pattern = re.compile(r'JOIN\s+(\w+)', re.IGNORECASE)
        
        tables_in_query = table_pattern.findall(query) + join_pattern.findall(query)
        
        for table in tables_in_query:
            if table not in self.schema_info:
                return False, f"Query references unknown table: {table}"
        
        return True, "Valid query"
    
    def process_query(self, user_query):
        """Process a natural language query"""
        start_time = time.time()

        # Add to conversation history
        if self.conversation_history and 'user' in self.conversation_history[-1] and not 'system' in self.conversation_history[-1]:
            # Update last turn if system hasn't responded yet
            self.conversation_history[-1]['user'] = user_query
        else:
            self.conversation_history.append({'user': user_query})
            
        # Prepare prompt and generate response
        prompt = self._prepare_prompt(user_query)
        # generated_text = self.nlp_pipeline(prompt)[0]['generated_text'] # if call local llm
        generated_text = self.model_engine.call_coder_llm(query=user_query, prompt=prompt)
        
        # Check if we need more information
        needs_clarification, clarification_question = self._needs_clarification(user_query, generated_text)
        if needs_clarification:
            response = f"{clarification_question}"
            self.conversation_history[-1]['system'] = response
            return response
            
        # Extract and validate SQL_QUERY
        sql_query = self._extract_sql_query(generated_text)
        if not sql_query:
            response = "I couldn't generate a valid SQL_QUERY. Can you rephrase your question?"
            self.conversation_history[-1]['system'] = response
            return response
            
        is_valid, validation_message = self._validate_sql(sql_query)
        if not is_valid:
            response = f"Generated SQL_QUERY is not valid: {validation_message}"
            self.conversation_history[-1]['system'] = response
            return response
            
        # Execute query and return results
        results = self.db.execute_query(sql_query)
        
        # Format response
        response = f"Generated SQL: {sql_query}\n\nResults:\n{results}"

        # Add timing info if in debug mode
        total_time = time.time() - start_time
        response += f"\n\nProcessing time: {total_time:.2f}s"

        self.conversation_history[-1]['system'] = response
        return response
    
    def add_document(self, file_path: str) -> str:
        """
        Add a new document to the knowledge base
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Result message
        """
        result = self.knowledge_base.add_document(file_path, self.doc_processor)
        
        self.knowledge_base.save()  # Update cache
        return result
# app/core/document_processor.py
import re
import os
import csv
import PyPDF2
import markdown
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    """Handles multiple document formats and creates embeddings for RAG"""
    def __init__(self, embedding_model="intfloat/multilingual-e5-small", embedding_dim=384):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = embedding_dim
        
        # Verify and potentially adjust embedding dimension
        actual_dim = self.embedding_model.get_sentence_embedding_dimension()
        if actual_dim != embedding_dim:
            print(f"Warning: Specified embedding dim {embedding_dim} does not match model's actual dim {actual_dim}. Using {actual_dim}.")
            self.embedding_dim = actual_dim
        
    def process_document(self, file_path):
        """Process document based on file extension"""
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        try:
            process_methods = {
                '.pdf': self._process_pdf,
                '.csv': self._process_csv,
                '.txt': self._process_txt,
                '.md': self._process_markdown,
                '.markdown': self._process_markdown
            }
            if file_extension in process_methods:
                return process_methods[file_extension](file_path)
            return f"Unsupported file format: {file_extension}"
        except Exception as e:
            raise RuntimeError(f"Error processing {file_path}: {e}")
    
    def _process_pdf(self, file_path):
        """Extract text from PDF file"""
        doc_text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    doc_text += page.extract_text() + " "
            return doc_text
        except Exception as e:
            return f"Error processing PDF: {e}"
    
    def _process_csv(self, file_path):
        """Extract information from CSV file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                csv_reader = csv.reader(file)
                headers = next(csv_reader, [])  # 避免 CSV 为空时报错
                
                # Get column descriptions and sample data
                desc_text = f"CSV File: {os.path.basename(file_path)}\nColumns: {', '.join(headers)}\n\n"
                
                # Add sample data (first 5 rows)
                desc_text += "Sample data:\n"
                for i, row in enumerate(csv_reader):
                    if i >= 5:  # Only take first 5 rows as samples
                        break
                    desc_text += f"Row {i+1}: {', '.join(row)}\n"
                
                return desc_text
        except Exception as e:
            return f"Error processing CSV: {e}"
    
    def _process_txt(self, file_path):
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            return f"Error processing TXT: {e}"
    
    def _process_markdown(self, file_path):
        """Extract text from Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
                # Convert markdown to plain text (simple approach)
                html = markdown.markdown(md_content)
                # Remove HTML tags
                plain_text = re.sub(r'<[^>]+>', ' ', html)
                return plain_text
        except Exception as e:
            return f"Error processing Markdown: {e}"
    
    def create_chunks(self, text, chunk_size=400, overlap=50):
        """Split text into overlapping chunks for better retrieval"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
                
        return chunks
    
    def create_embeddings(self, chunks, batch_size=32):
        """Create embeddings for text chunks with dimension control"""
        # Encode chunks
        embeddings = self.embedding_model.encode(chunks, batch_size=batch_size, convert_to_numpy=True)
        
        # Ensure correct dimension
        if embeddings.shape[1] != self.embedding_dim:
            print(f"Embedding shape {embeddings.shape} does not match expected dimension {self.embedding_dim}")
            # Truncate or pad embeddings if necessary
            if embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, :self.embedding_dim]
            else:
                pad_width = ((0, 0), (0, self.embedding_dim - embeddings.shape[1]))
                embeddings = np.pad(embeddings, pad_width, mode='constant')
        
        return embeddings.astype('float32')
    
# app/core/knowledge_base.py
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
        self.index = faiss.IndexFlatIP(embedding_dim)
        
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
            print(f"Embeddings shape: {embeddings.shape}") # Debugging
            
            # Normalize embeddings for better cosine similarity
            faiss.normalize_L2(embeddings)

            # Add to FAISS index
            if embeddings.shape[1] != self.embedding_dim:
                raise ValueError(f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.embedding_dim}")
            
            self.index.add(embeddings)

            process_time = time.time() - start_time
            return f"Added {len(chunks)} chunks from {source_name} in {process_time:.2f}s"
    
        except RuntimeError as e:
            return str(e)
        
    def retrieve_relevant(self, query, doc_processor, top_k=3):
        """Retrieve most relevant chunks for a query"""
        if not self.chunks:
            return []

        # Get more candidates than needed for MMR
        candidates_k = min(top_k * 3, len(self.chunks))
        
        # Check if query is in cache
        if query in self.query_cache:
            return self.query_cache[query]
        
        # Create query embedding and normalize
        query_embedding = doc_processor.embedding_model.encode([query])[0].reshape(1, -1).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        D, I = self.index.search(query_embedding, candidates_k)
        
        # Create results
        results = []
        for i, idx in enumerate(I[0]):
            if idx < len(self.chunks):
                results.append({
                    'text': self.chunks[idx],
                    'distance': D[0][i],  # Inner Product distance (higher is better)
                    'metadata': self.chunk_metadata[idx]
                })
                
                if len(results) == top_k:
                    break
        
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
        print(f"Query embedding shape: {query_embedding.shape}")  # 检查查询嵌入向量的形状``
        
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
        try:
            # Ensure cache directory exists
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # Full path for files
            data_path = os.path.join(self.cache_dir, 'kb_data.pkl')
            index_path = os.path.join(self.cache_dir, 'kb_faiss.index')
            
            # Save metadata
            with open(data_path, 'wb') as f:
                pickle.dump({
                    'chunks': self.chunks,
                    'chunk_metadata': self.chunk_metadata
                }, f)
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            print(f"Saved {len(self.chunks)} chunks to cache")
            return f"Saved {len(self.chunks)} chunks successfully"
        
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
            return f"Error saving knowledge base: {e}"
    
    def load(self):
        """Load knowledge base from disk"""
        try:
            # Full paths for files
            data_path = os.path.join(self.cache_dir, 'kb_data.pkl')
            index_path = os.path.join(self.cache_dir, 'kb_faiss.index')
            
            # Check if cache files exist
            if not (os.path.exists(data_path) and os.path.exists(index_path)):
                return "No cache found"
            
            # Load metadata
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.chunk_metadata = data['chunk_metadata']
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            return f"Loaded {len(self.chunks)} chunks from cache"
        
        except FileNotFoundError:
            return "No cache found"
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return f"Error loading knowledge base: {e}"