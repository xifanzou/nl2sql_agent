import os
import re
import time
from collections import Counter
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
        
        # Conversation history & summary
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
        results = self.knowledge_base.retrieve_with_mmr(query, self.doc_processor, top_k=5, diversity=0.35)
        
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

        for turn in self.conversation_history[-3:]:  # Last 3 turns for context
            user_text = turn.get("user", "Unknown User Input")
            system_text = turn.get("system", "No System Response")
            history_text += f"User: {user_text}\nSystem: {system_text}\n"

        # Detect if this is a follow-up question
        is_followup = self._detect_followup_question(user_query)
        
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
    
    def _detect_followup_question(self, query):
        """Detect if the current question is a follow-up to previous ones using LLM."""
        
        # Construct conversation context from the last few exchanges
        conversation_context = ""
        for turn in self.conversation_history[-3:]:  # Last 3 turns for context
            user_text = turn.get("user", "Unknown User Input")
            system_text = turn.get("system", "No System Response")
            conversation_context += f"User: {user_text}\nSystem: {system_text}\n"

        # Construct prompt to ask LLM
        prompt = f"""
        Given the conversation history below:

        {conversation_context}

        User's current query: {query}

        Is the user's current query a follow-up to the previous questions? Respond with 'yes' or 'no'.
        """

        # Call the LLM to evaluate whether the current query is a follow-up question
        followup_response = self.model_engine.call_helper(query=query, prompt=prompt)

        # Parse the LLM's response to determine follow-up
        if followup_response.strip().lower() == 'yes':
            return True
        else:
            return False
        
    def _handle_exploration_query(self, user_query):
        """
        Handle queries about available data and variables
        
        Returns:
            Detailed information about database schema and available data
        """
        relevant_knowledge = self._retrieve_relevant_knowledge(user_query)
        
        # Prepare prompt for data exploration
        exploration_prompt = f"""Database Schema Overview
        Database Schema: {self.schema_info}
        Relevant Knowledge: {relevant_knowledge}

        Answer user's question by analyzing relevant Database schema and relevant knowledge.
        Provide a clear, user-friendly explanation.

        Response Requirements:
        - Answer in the same language as the user’s query.
        - Use a clear and structured format (e.g., bullet points, tables).
        - Do not provide SQL queries.
        - Keep the response within 500 words.
        """
        
        # Generate exploration response
        exploration_response = self.model_engine.call_helper(
            query=user_query, 
            prompt=exploration_prompt)
        
        return exploration_response

    def _needs_explore(self, query):
        prompt = f"""
        Task: Determine if the user is asking a **general exploratory question** about available **data, indicators, metrics, categories, or entities** in a database.
        ### **Exploratory Question Examples** (Answer 'yes' if similar):
        - "What metrics can I query?"
        - "What data is available?"
        - "What kinds of failures are recorded?"
        - "What types of vehicles exist?"
        - "What can I analyze in this dataset?"
        - "Tell me about the available fields in the database."
        
        ### **Non-Exploratory Questions** (Answer 'no' if similar):
        - "What is the failure rate of vehicle type A?"
        - "How many failures happened last month?"
        - "Show me the top 10 most frequent failure types."
        - "What is the average speed of all vehicles?"
        - "Retrieve data for vehicle ID 1234."
        
        **Response Format:**  
        - Reply with only `'yes'` if the query is exploratory.  
        - Reply with only `'no'` if the query is specific.  
        - No extra text, explanations, or formatting.

        User Query: {query}
        """
        response = self.model_engine.call_helper(query=query, prompt=prompt)
        return response
        
    def _finalize_analysis(self, results, user_query):
        # Call helper_llm for natural language analysis of results
        analysis_prompt = f"""
        Given the user's query history: {self.conversation_history}
        The SQL query results: {results}
        The most recent user query: {user_query}

        Please provide a natural language analysis of the SQL results and a general summary that incorporates all relevant user inputs.
        """
        analysis_response = self.model_engine.call_llm(
            query=user_query, 
            prompt=analysis_prompt)
        
        return analysis_response
    
    def _needs_clarification(self, query, generated_text):
        """Determine if we need more information from the user"""
        if "请提供具体信息:" in generated_text:
            return True, generated_text.split("请提供具体信息:")[1].strip()
        elif "引导提问:" in generated_text:
            return True, generated_text.split("引导提问:")[1].strip()
        return False, ""
    
    def _extract_sql_query(self, generated_text):
        """
        Extract SQL_QUERY from generated text.
        Handles cases where SQL is wrapped in Markdown or followed by extra content.
        """
        if "SQL_QUERY:" not in generated_text:
            return None
        raw_sql = generated_text.split("SQL_QUERY:")[1].strip()
        clean_sql = re.sub(r"^```sql\s*|\s*```$", "", raw_sql, flags=re.MULTILINE).strip()

        sql_match = re.search(r"^(.*?);", clean_sql, re.DOTALL)
        if sql_match:
            final_sql = sql_match.group(1).strip() + ";"
        else:
            final_sql = clean_sql.strip()

        if not final_sql.endswith(";"):
            final_sql += ";"

        return final_sql
    
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
        
        _is_exploration = self._needs_explore(user_query)
        if 'yes' in _is_exploration:
            exploration_response = self._handle_exploration_query(user_query)
            self.conversation_history[-1]['system'] = exploration_response
            return exploration_response
            
        # Prepare prompt and generate response
        prompt = self._prepare_prompt(user_query)
        # generated_text = self.nlp_pipeline(prompt)[0]['generated_text'] # if call local llm
        generated_text = self.model_engine.call_coder(query=user_query, prompt=prompt)
        
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
        
        # Combine the SQL response with the analysis
        analysis_response = self._finalize_analysis(results, user_query)
        response += f"\n\nAnalysis:\n\n{analysis_response}"

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