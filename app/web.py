import streamlit as st
import os
from dotenv import load_dotenv
from chatbot.nl2sql import NL2SQLChatbot

# Load environment variables
load_dotenv()

# Initialize the chatbot
@st.cache_resource
def initialize_chatbot():
    return NL2SQLChatbot(
        documents_dir="./data"  # Directory containing documentation files
    )

def main():
    st.set_page_config(
        page_title="NL2SQL Assistant",
        page_icon="💬",
        layout="wide",
    )
    
    # Initialize session state for chat history if it doesn't exist
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Get or initialize the chatbot
    chatbot = initialize_chatbot()
    
    # App header
    st.title("💬 NL2SQL Assistant")
    st.markdown("""
    Ask questions about your data in natural language. 
    I'll translate them to SQL and return the results!
    """)
    
    # Sidebar for document upload and settings
    with st.sidebar:
        st.header("📁 Document Management")
        
        uploaded_file = st.file_uploader("Upload a document to the knowledge base", 
                                         type=["pdf", "txt", "csv", "md", "markdown"])
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            file_path = os.path.join("./data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Add to knowledge base
            with st.spinner(f"Processing {uploaded_file.name}..."):
                result = chatbot.add_document(file_path)
                st.success(f"Document processed: {result}")
    
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)
    
    # Chat input
    if prompt := st.chat_input("Ask about your data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatbot.process_query(prompt)
                
            # Format SQL code if present in the response
            if "Generated SQL:" in response:
                parts = response.split("Generated SQL:", 1)
                prefix = parts[0] if parts[0] else ""
                
                sql_and_rest = parts[1].split("\n\n", 1)
                sql = sql_and_rest[0].strip()
                rest = sql_and_rest[1] if len(sql_and_rest) > 1 else ""
                
                # Display with formatting
                if prefix:
                    st.markdown(prefix)
                st.markdown("**Generated SQL:**")
                st.code(sql, language="sql")
                if rest:
                    st.markdown(rest)
            else:
                # Just display the regular response
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()