import re
import os
import csv
import PyPDF2
import markdown
from sentence_transformers import SentenceTransformer

class DocumentProcessor:
    """Handles multiple document formats and creates embeddings for RAG"""
    def __init__(self, embedding_model="intfloat/multilingual-e5-small", embedding_dim=384):
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = embedding_dim
        
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
        """Create embeddings for text chunks"""
        return self.embedding_model.encode(chunks, batch_size=batch_size, convert_to_numpy=True)
    