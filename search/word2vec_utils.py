import os
import numpy as np
from gensim.models import Word2Vec
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Word2VecModel:
    def __init__(self, model_path=None, vector_cache_path=None):
        self.model_path = model_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'models', 'word2vec_model.model')
        self.vector_cache_path = vector_cache_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'models', 'vector_cache.pkl')
        self.model = None
        self.vector_cache = {}  # Cache for document vectors
        self.load_or_create_model()
        self.load_vector_cache()
        
    def load_or_create_model(self):
        """Load existing model or create a new one if it doesn't exist"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            if os.path.exists(self.model_path):
                logging.info(f"Loading Word2Vec model from {self.model_path}")
                self.model = Word2Vec.load(self.model_path)
                logging.info(f"Model loaded with vocabulary size: {len(self.model.wv.key_to_index)}")
            else:
                logging.info("Creating new Word2Vec model")
                self.model = Word2Vec(
                    vector_size=300,  
                    window=10,        
                    min_count=2,      
                    workers=8,        
                    sg=1              
                )
                # Save the empty model
                self.model.save(self.model_path)
                logging.info("Created new Word2Vec model")
        except Exception as e:
            logging.error(f"Error loading/creating model: {e}")
            # Create a basic model as fallback
            self.model = Word2Vec(
                vector_size=300,
                window=10,
                min_count=2,
                workers=8,
                sg=1  
            )
    
    def load_vector_cache(self):
        """Load cached document vectors if available"""
        try:
            if os.path.exists(self.vector_cache_path):
                with open(self.vector_cache_path, 'rb') as f:
                    self.vector_cache = pickle.load(f)
                logging.info(f"Vector cache loaded with {len(self.vector_cache)} entries")
            else:
                logging.info("No vector cache found, creating new cache")
                self.vector_cache = {}
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.vector_cache_path), exist_ok=True)
                # Save empty cache
                with open(self.vector_cache_path, 'wb') as f:
                    pickle.dump(self.vector_cache, f)
        except Exception as e:
            logging.error(f"Error loading vector cache: {e}")
            self.vector_cache = {}
    
    def save_vector_cache(self):
        """Save document vectors cache to disk"""
        try:
            os.makedirs(os.path.dirname(self.vector_cache_path), exist_ok=True)
            with open(self.vector_cache_path, 'wb') as f:
                pickle.dump(self.vector_cache, f)
            logging.info(f"Vector cache saved with {len(self.vector_cache)} entries")
        except Exception as e:
            logging.error(f"Error saving vector cache: {e}")
    
    def train_model(self, sentences, force_train=False):
        """Train the model with new sentences"""
        if not sentences:
            logging.info("No sentences provided for training, skipping")
            return False
            
        # Skip training if we already have a trained model and force_train is False
        if self.model.wv.key_to_index and not force_train:
            logging.info("Model already trained and force_train=False, skipping training")
            return False
            
        logging.info(f"Training Word2Vec model with {len(sentences)} documents")
        
        if not self.model.wv.key_to_index:
            # If vocabulary is empty, build it first
            logging.info("Building vocabulary from scratch")
            self.model.build_vocab(sentences)
        else:
            # Update vocabulary with new sentences
            logging.info("Updating existing vocabulary")
            self.model.build_vocab(sentences, update=True)
        
        # Train the model
        logging.info("Training Word2Vec model")
        self.model.train(
            sentences,
            total_examples=self.model.corpus_count,
            epochs=5
        )
        
        # Save the updated model
        logging.info(f"Saving model to {self.model_path}")
        self.model.save(self.model_path)
        logging.info(f"Model trained and saved with vocabulary size: {len(self.model.wv.key_to_index)}")
        
        # Clear vector cache after training
        self.vector_cache = {}
        self.save_vector_cache()
        return True
    
    @lru_cache(maxsize=1000)
    def get_word_vector_cached(self, word):
        """Get vector for a word with caching"""
        if word in self.model.wv:
            return self.model.wv[word]
        return np.zeros(self.model.vector_size)
    
    def get_document_vector(self, doc_id, tokens):
        """Calculate document vector from tokens with caching"""
        # Return cached vector if available
        if doc_id in self.vector_cache:
            return self.vector_cache[doc_id]
        
        if not tokens:
            return np.zeros(self.model.vector_size)
        
        # Calculate vector
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.get_word_vector_cached(token))
        
        if not vectors:
            result = np.zeros(self.model.vector_size)
        else:
            result = np.mean(vectors, axis=0)
        
        # Cache the result
        self.vector_cache[doc_id] = result
        return result
    
    def get_query_vector(self, tokens):
        """Calculate query vector from tokens"""
        if not tokens:
            return np.zeros(self.model.vector_size)
        
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.get_word_vector_cached(token))
        
        if not vectors:
            return np.zeros(self.model.vector_size)
        
        return np.mean(vectors, axis=0)
    
    def compute_similarity(self, vec1, vec2):
        """Compute cosine similarity between two vectors"""
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            return 0.0
        
        return dot_product / (norm_vec1 * norm_vec2)
    
    def compute_similarities_parallel(self, query_vector, doc_vectors):
        """Compute similarities in parallel"""
        with ThreadPoolExecutor(max_workers=8) as executor:
            similarities = list(executor.map(
                lambda doc_vector: self.compute_similarity(query_vector, doc_vector[1]),
                doc_vectors.items()
            ))
        return similarities
    
    def get_model_vectors(self, words):
        """Get vectors for a list of words (for display purposes)"""
        result = {}
        for word in words:
            if word in self.model.wv:
                # Get all dimensions for display
                vec = self.model.wv[word].tolist()
                result[word] = vec
        return result
    
    def clear_document_vector(self, doc_id):
        """Remove a document vector from cache when document is deleted"""
        if doc_id in self.vector_cache:
            del self.vector_cache[doc_id]
            self.save_vector_cache()
