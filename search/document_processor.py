import os
import re
import PyPDF2
from django.conf import settings
import nltk
from nltk.tokenize import word_tokenize
import string
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from functools import lru_cache

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentProcessor:
    def __init__(self):
        self.punctuation = set(string.punctuation)
        self.document_cache = {}  # Cache for document text and tokens
        
    @lru_cache(maxsize=100)
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file with caching"""
        try:
            # Check if already in cache
            if pdf_path in self.document_cache:
                return self.document_cache[pdf_path]['text']
                
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                
                # Store in cache
                if pdf_path not in self.document_cache:
                    self.document_cache[pdf_path] = {}
                self.document_cache[pdf_path]['text'] = text
                
                return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    @lru_cache(maxsize=1000)
    def preprocess_text(self, text):
        """Preprocess text: minimal preprocessing for semantic models"""
        if not text:
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Only filter non-alphabetic tokens for basic cleaning
        # No stopword removal (TF-IDF related concept)
        tokens = [token for token in tokens if token.isalpha()]
        
        return tokens
    
    def get_document_tokens(self, document_path):
        """Get tokens from a document with caching"""
        # Check if already in cache
        if document_path in self.document_cache and 'tokens' in self.document_cache[document_path]:
            return self.document_cache[document_path]['tokens'], self.document_cache[document_path]['text']
            
        text = self.extract_text_from_pdf(document_path)
        tokens = self.preprocess_text(text)
        
        # Store in cache
        if document_path not in self.document_cache:
            self.document_cache[document_path] = {}
        self.document_cache[document_path]['tokens'] = tokens
        self.document_cache[document_path]['text'] = text
        
        return tokens, text
    
    def get_all_documents_tokens(self, documents):
        """Get tokens for all documents in parallel"""
        document_paths = []
        for document in documents:
            full_path = document.document_path
            if not os.path.isabs(full_path):
                # If path is relative, make it absolute
                full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
            document_paths.append((document.id, full_path))
        
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(
                lambda doc: (doc[0], self.get_document_tokens(doc[1])[0]),
                document_paths
            ))
        
        # Convert to format expected by caller
        all_tokens = [tokens for _, tokens in results]
        return all_tokens
    
    def clear_document_cache(self, document_path):
        """Clear document from cache when deleted"""
        if document_path in self.document_cache:
            del self.document_cache[document_path]