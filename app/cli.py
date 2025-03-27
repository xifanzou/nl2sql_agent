import sys, os
from chatbot.nl2sql import NL2SQLChatbot
# Ensure Python can find the 'core' module inside 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "chatbot")))

absolute_path = os.path.abspath("data")
print("absolute_path: ", absolute_path)


# Initialize the chatbot
chatbot = NL2SQLChatbot(
    # model_name="google/flan-t5-small",  # Small LLM from Hugging Face
    documents_dir=absolute_path  # Directory containing all documentation files
)

def chat_loop():
    print("NL2SQL Chatbot initialized. Type 'exit/退出' to quit.")
    print("Special commands:")
    print("  - 'add_doc [path]': Add a new document to the knowledge base")
    
    while True:
        try:
            user_input = input("\nYour query: ").strip()
            if user_input.lower() == 'exit' or user_input.lower() == "退出":
                print("Exiting chatbot...")
                sys.exit(0)  # 确保退出程序
            
            if user_input.startswith('add_doc '):
                doc_path = user_input[8:].strip()
                result = chatbot.add_document(doc_path)
                print(f"\nResult: {result}\n")
                continue
            
            response = chatbot.process_query(user_input)
            print(f"\nChatbot: {response}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nUser interrupted. Exiting chatbot...")
            sys.exit(0)

if __name__ == "__main__":
    chat_loop()