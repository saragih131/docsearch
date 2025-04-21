import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
import datetime
from django.urls import reverse
import io
import base64
import time
import logging
from django.conf import settings
from .models import Document, Result
from .word2vec_utils import Word2VecModel
from .document_processor import DocumentProcessor
from .ocr_processor import OCRProcessor
from .transformer_utils import TransformerEmbeddings
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
from concurrent.futures import ThreadPoolExecutor
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from collections import defaultdict

# Global search engine instance
_search_engine = None

def initialize_search_engine():
    """Initialize the search engine at Django startup"""
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine()
        _search_engine.build_index()
        logging.info("Search engine initialized and index built")
    return _search_engine

def get_search_engine():
    """Get the global search engine instance"""
    global _search_engine
    if _search_engine is None:
        _search_engine = initialize_search_engine()
    return _search_engine

class SearchEngine:
    def __init__(self):
        self.word2vec_model = Word2VecModel()
        self.document_processor = DocumentProcessor()
        self.ocr_processor = OCRProcessor()
        self.transformer = TransformerEmbeddings()
        self.document_vectors = {}
        self.index_built = False
        self._last_search_data = None
    
    def build_index(self, force=False):
        """Build search index for faster retrieval"""
        if self.index_built and not force:
            logging.info("Index already built, skipping...")
            return
            
        start_time = time.time()
        logging.info("Building search index...")
        
        try:
            # Get all documents
            documents = Document.objects.all()
            
            # If no documents, mark index as built and return
            if not documents.exists():
                self.index_built = True
                logging.info("No documents found, index marked as built")
                return
                
            # Get all document tokens for training
            all_document_tokens = self.document_processor.get_all_documents_tokens(documents)
            
            # Train Word2Vec model with all documents
            self.word2vec_model.train_model(all_document_tokens, force_train=force)
            
            # Pre-compute document vectors
            doc_vectors = {}
            doc_texts = {}
            
            for document in documents:
                # Get document path
                full_path = document.document_path
                if not os.path.isabs(full_path):
                    full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
                
                # Get document tokens and text
                doc_tokens, doc_text = self.document_processor.get_document_tokens(full_path)
                
                # Store document text for transformer
                doc_texts[document.id] = doc_text
                
                # Get document vector
                doc_vector = self.word2vec_model.get_document_vector(document.id, doc_tokens)
                doc_vectors[document.id] = doc_vector
            
            # Save document vectors
            self.document_vectors = doc_vectors
            self.word2vec_model.save_vector_cache()
            
            # Pre-compute transformer embeddings
            if documents:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    for document in documents:
                        executor.submit(
                            self.transformer.get_document_embedding,
                            document.id,
                            doc_texts.get(document.id, "")
                        )
                self.transformer.save_cache()
            
            self.index_built = True
            logging.info(f"Search index built in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logging.error(f"Error building search index: {e}")
            # Don't set index_built = True if there's an error
    
    def search(self, ocr_text, ocr_id=None):
        """Search documents based on OCR text"""
        start_time = time.time()
        
        # Build index if not already built
        if not self.index_built:
            self.build_index()
        
        try:
            # Get all documents
            documents = Document.objects.all()
            
            # Preprocess OCR text
            ocr_tokens = self.document_processor.preprocess_text(ocr_text)
            
            # If no tokens after preprocessing, return empty results
            if not ocr_tokens:
                return [], None
            
            # Get document texts for transformer filtering
            doc_texts = {}
            for document in documents:
                # Get document path
                full_path = document.document_path
                if not os.path.isabs(full_path):
                    full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
                
                # Get document text
                _, doc_text = self.document_processor.get_document_tokens(full_path)
                doc_texts[document.id] = doc_text
            
            # Use transformer for initial filtering
            filtered_indices = self.transformer.filter_documents(ocr_text, documents, doc_texts)
            filtered_documents = [documents[i] for i in filtered_indices]
            
            logging.info(f"Transformer filtering: {len(filtered_documents)}/{len(documents)} documents")
            
            # Get query vector
            query_vector = self.word2vec_model.get_query_vector(ocr_tokens)
            
            # Calculate similarity for each filtered document
            results = []
            doc_vectors = {}
            
            # Get document vectors from cache when possible
            for document in filtered_documents:
                if document.id in self.document_vectors:
                    doc_vectors[document.id] = self.document_vectors[document.id]
                else:
                    # Get document path
                    full_path = document.document_path
                    if not os.path.isabs(full_path):
                        full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
                    
                    # Get document tokens
                    doc_tokens, _ = self.document_processor.get_document_tokens(full_path)
                    
                    # Get document vector
                    doc_vectors[document.id] = self.word2vec_model.get_document_vector(document.id, doc_tokens)
            
            # Calculate similarities in parallel
            similarities = []
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {}
                for document in filtered_documents:
                    futures[document.id] = executor.submit(
                        self.word2vec_model.compute_similarity,
                        query_vector,
                        doc_vectors[document.id]
                    )
                
                # Get results and create Result objects
                for document in filtered_documents:
                    similarity = futures[document.id].result()
                    
                    # Determine relevance category
                    if similarity >= 0.8:
                        category = "Sangat Relevan"
                    elif similarity >= 0.6:
                        category = "Relevan"
                    elif similarity >= 0.4:
                        category = "Cukup Relevan"
                    elif similarity >= 0.2:
                        category = "Sedikit Relevan"
                    else:
                        category = "Tidak Relevan"
                    
                    # Get model vectors for display (sample of tokens)
                    display_tokens = set(ocr_tokens[:10] + self.document_processor.get_document_tokens(
                        os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(document.document_path))
                    )[0][:10])
                    model_vectors = self.word2vec_model.get_model_vectors(display_tokens)
                    
                    # Format vectors for display - show all dimensions
                    model_vectors_str = ""
                    for word, vec in model_vectors.items():
                        if word in self.word2vec_model.model.wv:
                            # Show all dimensions
                            full_vec = self.word2vec_model.model.wv[word].tolist()
                            model_vectors_str += f'"{word}" → {full_vec}\n'
                    
                    # Create result record
                    result = Result(
                        id=document.id,
                        judul_artikel=document.document_name,
                        ekstrak_teks=f"{len(doc_texts.get(document.id, ''))} karakter yang ditemukan",
                        tokenisasi=json.dumps(self.document_processor.get_document_tokens(
                            os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(document.document_path))
                        )[0]),  # Show all tokens
                        kueri_ocr=json.dumps(ocr_tokens),  # Show all tokens
                        pelatihan_model=model_vectors_str,
                        perhitungan_vektor_dokumen=str(doc_vectors[document.id].tolist()),  # Show all dimensions
                        perhitungan_kueri_ocr=str(query_vector.tolist()),  # Show all dimensions
                        perhitungan_kesamaan_kosinus=similarity,
                        keterangan=category
                    )
                    result.save()
                    
                    # Add to results list
                    results.append({
                        'id': document.id,
                        'title': document.document_name,
                        'path': document.document_path,
                        'similarity': similarity,
                        'category': category
                    })
            
            # Sort results by similarity (descending)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Calculate search time
            search_time = time.time() - start_time
            
            # Save data for evaluation later
            self._last_search_data = {
                'results': results.copy(),
                'ocr_tokens': ocr_tokens,
                'search_time': search_time,
                'doc_vectors': doc_vectors,
                'query_vector': query_vector
            }
            
            # Return results immediately without waiting for evaluation
            return results, None
        except Exception as e:
            logging.error(f"Error during search: {e}")
            return [], None

    # Fungsi untuk menjalankan evaluasi secara terpisah
    def run_evaluation_async(self):
        """Run evaluation asynchronously after search results are displayed"""
        if not hasattr(self, '_last_search_data') or not self._last_search_data:
            print("No search data available for evaluation")
            return
        
        data = self._last_search_data
        
        # Generate visualizations
        pca_path = self.generate_pca_visualization(data['ocr_tokens'], data['results'])
        umap_path = self.generate_umap_visualization(data['ocr_tokens'], data['results'])

        
        # 1. ANALISIS KUERI OCR
        print("\n===== ANALISIS KUERI OCR =====")
        print(f"Jumlah token dalam kueri: {len(data['ocr_tokens'])}")
        print("Token kueri:")
        print(data['ocr_tokens'][:20])
        
        # 2. KONFIGURASI MODEL WORD2VEC
        print("\n===== KONFIGURASI MODEL WORD2VEC =====")
        print(f"Algoritma: Skipgram (sg=1)")
        print(f"Dimensi vektor: {self.word2vec_model.model.vector_size}")
        print(f"Window size: {self.word2vec_model.model.window}")
        print(f"Min count: {self.word2vec_model.model.min_count}")
        print(f"Workers: {self.word2vec_model.model.workers}")
        print(f"Vocabulary size: {len(self.word2vec_model.model.wv.key_to_index):,}")
        
        # 3. VISUALISASI WORD2VEC
        print("\n===== VISUALISASI WORD2VEC =====")
        # Tampilkan path lengkap untuk visualisasi PCA
        pca_full_path = os.path.join(settings.BASE_DIR, 'static', 'images', 'pca_visualization.png')
        print("Visualisasi PCA:")
        print(pca_full_path)
        
        # Tampilkan path lengkap untuk visualisasi UMAP
        umap_full_path = os.path.join(settings.BASE_DIR, 'static', 'images', 'umap_visualization.png')
        print("\nVisualisasi UMAP:")
        print(umap_full_path)
        
        # 4. ANALISIS KINERJA PENCARIAN
        print("\n===== ANALISIS KINERJA PENCARIAN =====")
        # Ubah format waktu pencarian
        print(f"Waktu pencarian: {int(data['search_time'])} detik")
        top10_precision = sum(1 for r in data['results'][:10] if r['category'] in ['Sangat Relevan', 'Relevan', 'Cukup Relevan']) / min(10, len(data['results'])) if data['results'] else 0
        print(f"Top-10 Precision: {top10_precision:.4f}")
        
        # 5. EVALUASI KINERJA SISTEM PENCARIAN DOKUMEN
        print("\n===== EVALUASI KINERJA SISTEM PENCARIAN DOKUMEN =====")
        # Count documents by category
        categories = {
            'Sangat Relevan': 0,
            'Relevan': 0,
            'Cukup Relevan': 0,
            'Sedikit Relevan': 0,
            'Tidak Relevan': 0
        }
        
        for result in data['results']:
            categories[result['category']] += 1
        
        # Calculate precision, recall, and F1 score
        relevant_docs = categories['Sangat Relevan'] + categories['Relevan'] + categories['Cukup Relevan']
        total_docs = sum(categories.values())
        
        # Precision = relevant documents retrieved / total documents retrieved
        precision = relevant_docs / total_docs if total_docs > 0 else 0
        
        # For recall, we need to estimate the total relevant documents in the collection
        # Since we don't know the ground truth, we'll use a heuristic
        estimated_total_relevant = relevant_docs + (categories['Sedikit Relevan'] // 2)
        recall = relevant_docs / estimated_total_relevant if estimated_total_relevant > 0 else 0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Mean Average Precision (MAP)
        # For simplicity, we'll calculate a simplified version
        ap_sum = 0
        relevant_count = 0
        
        for i, result in enumerate(data['results'][:10], start=1):
            if result['category'] in ['Sangat Relevan', 'Relevan', 'Cukup Relevan']:
                relevant_count += 1
                ap_sum += relevant_count / i
        
        map_score = ap_sum / relevant_count if relevant_count > 0 else 0
        
        # Calculate Mean Reciprocal Rank (MRR)
        mrr = 0
        for i, result in enumerate(data['results'], start=1):
            if result['category'] in ['Sangat Relevan', 'Relevan']:
                mrr = 1 / i
                break
        
        # Calculate Normalized Discounted Cumulative Gain (NDCG)
        # First, create relevance scores (4 for Sangat Relevan, 3 for Relevan, etc.)
        relevance_scores = []
        for result in data['results'][:10]:  # Consider top 10 results
            if result['category'] == 'Sangat Relevan':
                relevance_scores.append(4)
            elif result['category'] == 'Relevan':
                relevance_scores.append(3)
            elif result['category'] == 'Cukup Relevan':
                relevance_scores.append(2)
            elif result['category'] == 'Sedikit Relevan':
                relevance_scores.append(1)
            else:
                relevance_scores.append(0)
        
        # Calculate DCG
        dcg = 0
        for i, score in enumerate(relevance_scores, 1):
            dcg += score / np.log2(i + 1)
        
        # Calculate ideal DCG (sorted relevance scores)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = 0
        for i, score in enumerate(ideal_relevance, 1):
            idcg += score / np.log2(i + 1)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        
        # Print metrics in the requested format
        print(f"Total dokumen: {total_docs}")
        print(f"Dokumen sangat relevan: {categories['Sangat Relevan']}")
        print(f"Dokumen relevan: {categories['Relevan']}")
        print(f"Dokumen cukup relevan: {categories['Cukup Relevan']}")
        print(f"Dokumen sedikit relevan: {categories['Sedikit Relevan']}")
        print(f"Dokumen tidak relevan: {categories['Tidak Relevan']}")
        print()
        print("Metrik Evaluasi:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Mean Average Precision (MAP): {map_score:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print(f"Normalized Discounted Cumulative Gain (NDCG@10): {ndcg:.4f}")
    
    def print_word2vec_evaluation(self):
        """Print Word2Vec model evaluation"""
        print("\n======== Evaluasi Model Word2Vec ========")
        print(f" Algoritma: Skipgram")
        print(f" Dimensi vektor: {self.word2vec_model.model.vector_size}")
        print(f" Window size: {self.word2vec_model.model.window}")
        print(f" Min count: {self.word2vec_model.model.min_count}")
        print(f" Workers: {self.word2vec_model.model.workers}")
        print(f" Vocabulary size: {len(self.word2vec_model.model.wv.key_to_index)}")
    
    def print_pca_visualization_info(self, pca_path):
        """Print PCA visualization information"""
        print("\n======== Visualisasi PCA untuk Word2Vec ========")
        full_path = os.path.join(settings.BASE_DIR, 'static', 'images', 'pca_visualization.png')
        print(f"Visualisasi PCA disimpan di: {full_path}")
    
    def print_umap_visualization_info(self, umap_path):
        """Print UMAP visualization information"""
        print("\n======== Visualisasi UMAP untuk Word2Vec ========")
        full_path = os.path.join(settings.BASE_DIR, 'static', 'images', 'umap_visualization.png')
        print(f"Visualisasi UMAP disimpan di: {full_path}")
    
    def print_ocr_query_analysis(self, query_tokens):
        """Print OCR query analysis"""
        print("\n======== Analisis Kueri OCR ========")
        print(f" Jumlah token dalam kueri: {len(query_tokens)}")
        print(" Kata-kata dalam kueri:")
        query_words_str = ", ".join([f'"{token}"' for token in query_tokens[:15]])
        if len(query_tokens) > 15:
            query_words_str += ", ..."
        print(query_words_str)
    
    def evaluate_search_performance(self, results, query_tokens, search_time):
        """Evaluate search performance and print metrics"""
        # Count documents by category
        categories = {
            'Sangat Relevan': 0,
            'Relevan': 0,
            'Cukup Relevan': 0,
            'Sedikit Relevan': 0,
            'Tidak Relevan': 0
        }
        
        for result in results:
            categories[result['category']] += 1
        
        # Calculate precision, recall, and F1 score
        relevant_docs = categories['Sangat Relevan'] + categories['Relevan'] + categories['Cukup Relevan']
        total_docs = sum(categories.values())
        
        # Precision = relevant documents retrieved / total documents retrieved
        precision = relevant_docs / total_docs if total_docs > 0 else 0
        
        # For recall, we need to estimate the total relevant documents in the collection
        # Since we don't know the ground truth, we'll use a heuristic
        estimated_total_relevant = relevant_docs + (categories['Sedikit Relevan'] // 2)
        recall = relevant_docs / estimated_total_relevant if estimated_total_relevant > 0 else 0
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Mean Average Precision (MAP)
        # For simplicity, we'll calculate a simplified version
        ap_sum = 0
        relevant_count = 0
        
        for i, result in enumerate(results[:10], start=1):
            if result['category'] in ['Sangat Relevan', 'Relevan', 'Cukup Relevan']:
                relevant_count += 1
                ap_sum += relevant_count / i
        
        map_score = ap_sum / relevant_count if relevant_count > 0 else 0
        
        # Calculate Mean Reciprocal Rank (MRR)
        mrr = 0
        for i, result in enumerate(results, start=1):
            if result['category'] in ['Sangat Relevan', 'Relevan']:
                mrr = 1 / i
                break
        
        # Generate precision-recall curve data
        y_true = [1 if r['category'] in ['Sangat Relevan', 'Relevan', 'Cukup Relevan'] else 0 for r in results]
        y_scores = [r['similarity'] for r in results]
        
        # Calculate Normalized Discounted Cumulative Gain (NDCG)
        # First, create relevance scores (4 for Sangat Relevan, 3 for Relevan, etc.)
        relevance_scores = []
        for result in results[:10]:  # Consider top 10 results
            if result['category'] == 'Sangat Relevan':
                relevance_scores.append(4)
            elif result['category'] == 'Relevan':
                relevance_scores.append(3)
            elif result['category'] == 'Cukup Relevan':
                relevance_scores.append(2)
            elif result['category'] == 'Sedikit Relevan':
                relevance_scores.append(1)
            else:
                relevance_scores.append(0)
        
        # Calculate DCG
        dcg = 0
        for i, score in enumerate(relevance_scores, 1):
            dcg += score / np.log2(i + 1)
        
        # Calculate ideal DCG (sorted relevance scores)
        ideal_relevance = sorted(relevance_scores, reverse=True)
        idcg = 0
        for i, score in enumerate(ideal_relevance, 1):
            idcg += score / np.log2(i + 1)
        
        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0
        
        # Print evaluation metrics
        print("\n======== Evaluasi Kinerja Sistem Pencarian Dokumen ========")
        print(f" Total dokumen: {total_docs}")
        print(f" Dokumen sangat relevan: {categories['Sangat Relevan']}")
        print(f" Dokumen relevan: {categories['Relevan']}")
        print(f" Dokumen cukup relevan: {categories['Cukup Relevan']}")
        print(f" Dokumen sedikit relevan: {categories['Sedikit Relevan']}")
        print(f" Dokumen tidak relevan: {categories['Tidak Relevan']}")
        print(f" Precision: {precision:.4f}")
        print(f" Recall: {recall:.4f}")
        print(f" F1 Score: {f1:.4f}")
        print(f" Mean Average Precision (MAP): {map_score:.4f}")
        print(f" Mean Reciprocal Rank (MRR): {mrr:.4f}")
        print(f" Normalized Discounted Cumulative Gain (NDCG@10): {ndcg:.4f}")
        print(f" Top-10 Precision: {sum(1 for r in results[:10] if r['category'] in ['Sangat Relevan', 'Relevan', 'Cukup Relevan']) / min(10, len(results)) if results else 0:.4f}")
        print(f" Waktu Pencarian: {int(search_time)} detik")
        
        return {
            'total_docs': total_docs,
            'categories': categories,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'map': map_score,
            'mrr': mrr,
            'ndcg': ndcg,
            'search_time': search_time,
            'top10_precision': sum(1 for r in results[:10] if r['category'] in ['Sangat Relevan', 'Relevan', 'Cukup Relevan']) / min(10, len(results)) if results else 0
        }

    def assign_word_to_category(self, word):
        """Assign a word to one of the 10 predefined categories based on semantic similarity"""
        # Define category keywords
        category_keywords = {
            "Transportasi": ["transportasi", "kendaraan", "mobil", "kereta", "bus", "jalan", "bandara", "pelabuhan", "terminal", "stasiun", "penerbangan", "kapal", "lalu lintas", "angkutan", "perjalanan"],
            "Pemerintahan": ["pemerintah", "presiden", "menteri", "gubernur", "bupati", "walikota", "kabinet", "kementerian", "birokrasi", "administrasi", "negara", "kebijakan", "pejabat", "aparatur", "daerah"],
            "Hukum & Regulasi": ["hukum", "undang", "peraturan", "regulasi", "pengadilan", "hakim", "jaksa", "advokat", "keadilan", "sanksi", "pelanggaran", "pidana", "perdata", "konstitusi", "legislasi"],
            "Pendidikan": ["pendidikan", "sekolah", "universitas", "akademik", "mahasiswa", "siswa", "guru", "dosen", "kurikulum", "pembelajaran", "pengajaran", "kampus", "riset", "penelitian", "ilmu"],
            "Infrastruktur & Proyek": ["infrastruktur", "proyek", "pembangunan", "konstruksi", "jembatan", "gedung", "jalan", "bendungan", "irigasi", "fasilitas", "sarana", "prasarana", "pengembangan", "investasi", "anggaran"],
            "Lingkungan & Bencana": ["lingkungan", "bencana", "alam", "ekosistem", "polusi", "pencemaran", "konservasi", "hutan", "banjir", "gempa", "tsunami", "kebakaran", "kekeringan", "iklim", "cuaca"],
            "Teknologi & Sistem": ["teknologi", "sistem", "digital", "komputer", "internet", "aplikasi", "software", "hardware", "jaringan", "data", "informasi", "inovasi", "elektronik", "otomasi", "platform"],
            "Politik & Tokoh Publik": ["politik", "partai", "pemilu", "demokrasi", "kampanye", "legislatif", "eksekutif", "parlemen", "tokoh", "figur", "publik", "pemimpin", "oposisi", "koalisi", "konstituen"],
            "Sosial & Masyarakat": ["sosial", "masyarakat", "komunitas", "budaya", "tradisi", "adat", "keluarga", "kesejahteraan", "bantuan", "kemiskinan", "kesehatan", "penduduk", "demografi", "interaksi", "kelompok"],
            "Ekonomi & Keuangan": ["ekonomi", "keuangan", "bisnis", "perdagangan", "investasi", "pasar", "saham", "bank", "kredit", "pajak", "anggaran", "inflasi", "pertumbuhan", "ekspor", "impor"]
        }
        
        # English translations and additional terms
        english_keywords = {
            "Transportasi": ["transportation", "vehicle", "car", "train", "bus", "road", "airport", "port", "terminal", "station", "flight", "ship", "traffic", "transport", "travel", "aviation"],
            "Pemerintahan": ["government", "president", "minister", "governor", "regent", "mayor", "cabinet", "ministry", "bureaucracy", "administration", "state", "policy", "official", "apparatus", "region", "public"],
            "Hukum & Regulasi": ["law", "act", "regulation", "court", "judge", "prosecutor", "advocate", "justice", "sanction", "violation", "criminal", "civil", "constitution", "legislation", "legal", "compliance"],
            "Pendidikan": ["education", "school", "university", "academic", "student", "teacher", "lecturer", "curriculum", "learning", "teaching", "campus", "research", "study", "science", "knowledge", "college"],
            "Infrastruktur & Proyek": ["infrastructure", "project", "development", "construction", "bridge", "building", "road", "dam", "irrigation", "facility", "means", "infrastructure", "development", "investment", "budget", "planning"],
            "Lingkungan & Bencana": ["environment", "disaster", "nature",   "budget", "planning"],
            "Lingkungan & Bencana": ["environment", "disaster", "nature", "ecosystem", "pollution", "contamination", "conservation", "forest", "flood", "earthquake", "tsunami", "fire", "drought", "climate", "weather", "green"],
            "Teknologi & Sistem": ["technology", "system", "digital", "computer", "internet", "application", "software", "hardware", "network", "data", "information", "innovation", "electronic", "automation", "platform", "tech"],
            "Politik & Tokoh Publik": ["politics", "party", "election", "democracy", "campaign", "legislative", "executive", "parliament", "figure", "public", "leader", "opposition", "coalition", "constituent", "political", "diplomat"],
            "Sosial & Masyarakat": ["social", "society", "community", "culture", "tradition", "custom", "family", "welfare", "aid", "poverty", "health", "population", "demographic", "interaction", "group", "people"],
            "Ekonomi & Keuangan": ["economy", "finance", "business", "trade", "investment", "market", "stock", "bank", "credit", "tax", "budget", "inflation", "growth", "export", "import", "economic", "financial", "commercial"]
        }
        
        # Combine Indonesian and English keywords
        for category, eng_words in english_keywords.items():
            category_keywords[category].extend(eng_words)
        
        # If the word is directly in a category's keywords, return that category
        word_lower = word.lower()
        for category, keywords in category_keywords.items():
            if word_lower in keywords:
                return category
        
        # If not found directly, try to find the most similar category based on word embeddings
        if word in self.word2vec_model.model.wv:
            word_vector = self.word2vec_model.model.wv[word]
            
            # Calculate similarity to each category (using average of keyword vectors)
            category_similarities = {}
            for category, keywords in category_keywords.items():
                # Get vectors for keywords that exist in the model
                keyword_vectors = []
                for keyword in keywords:
                    if keyword in self.word2vec_model.model.wv:
                        keyword_vectors.append(self.word2vec_model.model.wv[keyword])
                
                if keyword_vectors:
                    # Calculate average vector for this category
                    category_vector = np.mean(keyword_vectors, axis=0)
                    # Calculate cosine similarity
                    similarity = self.word2vec_model.compute_similarity(word_vector, category_vector)
                    category_similarities[category] = similarity
            
            # Return the category with highest similarity
            if category_similarities:
                return max(category_similarities.items(), key=lambda x: x[1])[0]
        
        # Fallback: assign based on simple string matching
        for category, keywords in category_keywords.items():
            for keyword in keywords:
                if keyword in word_lower or word_lower in keyword:
                    return category
        
        # If all else fails, return a default category based on first letter
        first_char = word_lower[0] if word_lower else 'a'
        categories = list(category_keywords.keys())
        return categories[ord(first_char) % len(categories)]

    def generate_pca_visualization(self, query_tokens, results):
        """Generate PCA visualization for Word2Vec vectors with semantic categorization"""
        try:
            # Create a directory for PCA visualizations if it doesn't exist
            pca_dir = os.path.join(settings.BASE_DIR, 'static', 'images')
            os.makedirs(pca_dir, exist_ok=True)
            
            # Use a fixed filename for PCA visualization
            pca_filename = "pca_visualization.png"
            pca_path = os.path.join(pca_dir, pca_filename)
            
            # Collect words for visualization
            words = set()
            
            # Add query tokens (up to 50)
            for token in query_tokens[:50]:
                if token in self.word2vec_model.model.wv:
                    words.add(token)
            
            # Add top words from relevant documents (up to 500 total words)
            for result in results[:15]:
                doc_id = result['id']
                doc = Document.objects.get(id=doc_id)
                
                # Get document path
                full_path = doc.document_path
                if not os.path.isabs(full_path):
                    full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
                
                # Get document tokens
                doc_tokens, _ = self.document_processor.get_document_tokens(full_path)
                
                # Add most frequent tokens
                for token in doc_tokens[:50]:
                    if token in self.word2vec_model.model.wv and len(words) < 500:
                        words.add(token)
            
            # If we don't have enough words, add some from the vocabulary
            if len(words) < 100 and len(self.word2vec_model.model.wv.key_to_index) > 0:
                # Add some common words from the vocabulary
                common_words = list(self.word2vec_model.model.wv.key_to_index.keys())[:500]
                for word in common_words:
                    if len(words) < 500:
                        words.add(word)
            
            # If we have words, create visualization
            if words:
                # Convert words to list for consistent ordering
                words_list = list(words)
                
                # Get word vectors
                word_vectors = np.array([self.word2vec_model.model.wv[word] for word in words_list])
                
                # Apply PCA to reduce to 2 dimensions
                pca = PCA(n_components=2)
                result = pca.fit_transform(word_vectors)
                
                # Define categories
                categories = [
                    "Transportasi",
                    "Pemerintahan",
                    "Hukum & Regulasi",
                    "Pendidikan",
                    "Infrastruktur & Proyek",
                    "Lingkungan & Bencana",
                    "Teknologi & Sistem",
                    "Politik & Tokoh Publik",
                    "Sosial & Masyarakat",
                    "Ekonomi & Keuangan"
                ]
                
                # Assign each word to a category based on semantic meaning
                word_categories = {}
                for word in words_list:
                    category = self.assign_word_to_category(word)
                    word_categories[word] = category
                
                # Group words by category
                category_words = defaultdict(list)
                category_indices = defaultdict(list)
                
                for i, word in enumerate(words_list):
                    category = word_categories[word]
                    category_words[category].append(word)
                    category_indices[category].append(i)
                
                # Define a colorful colormap
                colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                category_colors = {categories[i]: colors[i] for i in range(len(categories))}
                
                # Create figure with white background - larger size for more words
                plt.figure(figsize=(16, 14), facecolor='white')
                
                # Plot each category with a different color
                for category in categories:
                    if category in category_indices and category_indices[category]:
                        indices = category_indices[category]
                        plt.scatter(
                            result[indices, 0], 
                            result[indices, 1], 
                            c=[category_colors[category]], 
                            label=category,
                            alpha=0.7,
                            s=50,
                            edgecolors='w'
                        )
                
                # Add labels for each point
                for i, word in enumerate(words_list):
                    plt.annotate(
                        word, 
                        xy=(result[i, 0], result[i, 1]), 
                        xytext=(3, 1),
                        textcoords='offset points',
                        fontsize=7,
                        fontweight='normal',
                        color='black'
                    )
                
                # Add title and labels with better styling
                plt.title('PCA Visualization of Word2Vec Vectors', fontsize=18, fontweight='bold', pad=20)
                plt.xlabel('Principal Component 1', fontsize=14, fontweight='bold')
                plt.ylabel('Principal Component 2', fontsize=14, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.3)
                
                # Add a border
                plt.gca().spines['top'].set_visible(True)
                plt.gca().spines['right'].set_visible(True)
                plt.gca().spines['bottom'].set_visible(True)
                plt.gca().spines['left'].set_visible(True)
                
                # Add legend with better styling
                legend = plt.legend(
                    title="Kategori",
                    title_fontsize=12,
                    fontsize=10,
                    loc='best',
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='black'
                )
                
                # Improve tick labels
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                
                # Adjust layout and save with high quality
                plt.tight_layout()
                plt.savefig(pca_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                # Create an empty visualization if no words available
                plt.figure(figsize=(16, 14), facecolor='white')
                plt.title('PCA Visualization (No Words Available)', fontsize=18, fontweight='bold')
                plt.xlabel('Principal Component 1', fontsize=14, fontweight='bold')
                plt.ylabel('Principal Component 2', fontsize=14, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig(pca_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Return the relative path for the template
            return os.path.join('images', pca_filename)
        except Exception as e:
            logging.error(f"Error generating PCA visualization: {e}")
            print(f"Error generating PCA visualization: {e}")
            
            # Create an error visualization
            try:
                plt.figure(figsize=(16, 14), facecolor='white')
                plt.title(f'PCA Visualization Error: {str(e)}', fontsize=18, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(pca_dir, pca_filename), dpi=300, bbox_inches='tight')
                plt.close()
            except:
                pass
                
            return os.path.join('images', pca_filename)
            
    def generate_umap_visualization(self, query_tokens, results):
        """Generate UMAP visualization for Word2Vec vectors with semantic categorization"""
        try:
            # Create a directory for UMAP visualizations if it doesn't exist
            umap_dir = os.path.join(settings.BASE_DIR, 'static', 'images')
            os.makedirs(umap_dir, exist_ok=True)
            
            # Use a fixed filename for UMAP visualization
            umap_filename = "umap_visualization.png"
            umap_path = os.path.join(umap_dir, umap_filename)
            
            # Collect words for visualization
            words = set()
            
            # Add query tokens (up to 50)
            for token in query_tokens[:50]:
                if token in self.word2vec_model.model.wv:
                    words.add(token)
            
            # Add top words from relevant documents (up to 500 total words)
            for result in results[:15]:
                doc_id = result['id']
                doc = Document.objects.get(id=doc_id)
                
                # Get document path
                full_path = doc.document_path
                if not os.path.isabs(full_path):
                    full_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(full_path))
                
                # Get document tokens
                doc_tokens, _ = self.document_processor.get_document_tokens(full_path)
                
                # Add most frequent tokens
                for token in doc_tokens[:50]:
                    if token in self.word2vec_model.model.wv and len(words) < 500:
                        words.add(token)
            
            # If we don't have enough words, add some from the vocabulary
            if len(words) < 100 and len(self.word2vec_model.model.wv.key_to_index) > 0:
                # Add some common words from the vocabulary
                common_words = list(self.word2vec_model.model.wv.key_to_index.keys())[:500]
                for word in common_words:
                    if len(words) < 500:
                        words.add(word)
            
            # If we have words, create visualization
            if words:
                # Convert words to list for consistent ordering
                words_list = list(words)
                
                # Get word vectors as numpy array
                word_vectors = np.array([self.word2vec_model.model.wv[word] for word in words_list])
                
                # Define categories
                categories = [
                    "Transportasi",
                    "Pemerintahan",
                    "Hukum & Regulasi",
                    "Pendidikan",
                    "Infrastruktur & Proyek",
                    "Lingkungan & Bencana",
                    "Teknologi & Sistem",
                    "Politik & Tokoh Publik",
                    "Sosial & Masyarakat",
                    "Ekonomi & Keuangan"
                ]
                
                # Assign each word to a category based on semantic meaning
                word_categories = {}
                for word in words_list:
                    category = self.assign_word_to_category(word)
                    word_categories[word] = category
                
                # Group words by category
                category_words = defaultdict(list)
                category_indices = defaultdict(list)
                
                for i, word in enumerate(words_list):
                    category = word_categories[word]
                    category_words[category].append(word)
                    category_indices[category].append(i)
                
                # Configure UMAP for handling more words
                n_neighbors = min(30, max(5, len(words_list) // 10))
                min_dist = 0.05
                
                # Apply UMAP to reduce to 2 dimensions
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=2,
                    random_state=42
                )
                embedding = reducer.fit_transform(word_vectors)
                
                # Define a colorful colormap
                colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
                category_colors = {categories[i]: colors[i] for i in range(len(categories))}
                
                # Create figure with white background - larger size for more words
                plt.figure(figsize=(16, 14), facecolor='white')
                
                # Plot each category with a different color
                for category in categories:
                    if category in category_indices and category_indices[category]:
                        indices = category_indices[category]
                        plt.scatter(
                            embedding[indices, 0], 
                            embedding[indices, 1], 
                            c=[category_colors[category]], 
                            label=category,
                            alpha=0.7,
                            s=50,
                            edgecolors='w'
                        )
                
                # Add labels for each point
                for i, word in enumerate(words_list):
                    plt.annotate(
                        word, 
                        xy=(embedding[i, 0], embedding[i, 1]), 
                        xytext=(3, 1),
                        textcoords='offset points',
                        fontsize=7,
                        fontweight='normal',
                        color='black'
                    )
                
                # Add title and labels with better styling
                plt.title('UMAP Visualization of Word2Vec Vectors', fontsize=18, fontweight='bold', pad=20)
                plt.xlabel('UMAP Component 1', fontsize=14, fontweight='bold')
                plt.ylabel('UMAP Component 2', fontsize=14, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.3)
                
                # Add a border
                plt.gca().spines['top'].set_visible(True)
                plt.gca().spines['right'].set_visible(True)
                plt.gca().spines['bottom'].set_visible(True)
                plt.gca().spines['left'].set_visible(True)
                
                # Add legend with better styling
                legend = plt.legend(
                    title="Kategori",
                    title_fontsize=12,
                    fontsize=10,
                    loc='best',
                    frameon=True,
                    framealpha=0.9,
                    edgecolor='black'
                )
                
                # Improve tick labels
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                
                # Adjust layout and save with high quality
                plt.tight_layout()
                plt.savefig(umap_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                # Create an empty visualization if no words available
                plt.figure(figsize=(16, 14), facecolor='white')
                plt.title('UMAP Visualization (No Words Available)', fontsize=18, fontweight='bold')
                plt.xlabel('UMAP Component 1', fontsize=14, fontweight='bold')
                plt.ylabel('UMAP Component 2', fontsize=14, fontweight='bold')
                plt.grid(True, linestyle='--', alpha=0.3)
                plt.tight_layout()
                plt.savefig(umap_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Return the relative path for the template
            return os.path.join('images', umap_filename)
        except Exception as e:
            logging.error(f"Error generating UMAP visualization: {e}")
            print(f"Error generating UMAP visualization: {e}")
            
            # Create an error visualization
            try:
                plt.figure(figsize=(16, 14), facecolor='white')
                plt.title(f'UMAP Visualization Error: {str(e)}', fontsize=18, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(umap_dir, umap_filename), dpi=300, bbox_inches='tight')
                plt.close()
            except:
                pass
                
            return os.path.join('images', umap_filename)