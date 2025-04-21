import os
import shutil
import datetime
import json
import time
import hashlib
from django.shortcuts import render, redirect
from django.http import JsonResponse, HttpResponse, FileResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.db import transaction
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from .models import Document, Result
from .search_engine import get_search_engine
from .ocr_processor import OCRProcessor
import logging
import threading

# Set maximum upload size to 200 KB
MAX_UPLOAD_SIZE = 200 * 1024  # 200 KB in bytes

def index(request):
    """Main page view"""
    documents = Document.objects.all().order_by('id')
    return render(request, 'index.html', {'documents': documents})

@csrf_exempt
def search_documents(request):
    """Handle document search"""
    if request.method == 'POST' and request.FILES.get('image'):
        # Get uploaded image
        image_file = request.FILES['image']
        
        # Process OCR
        ocr_processor = OCRProcessor()
        ocr_text = ocr_processor.process_image(image_file)
        
        # Clear previous results to ensure fresh results for each search
        Result.objects.all().delete()
        
        # Check if the image contains text
        if ocr_text == "GAMBAR TIDAK BERISI TEKS":
            # Display notification and stop processing
            messages.error(request, 'GAMBAR TIDAK BERISI TEKS')
            # Return to index without any search results or OCR text
            return redirect('index')
        
        # If we get here, the image contains text, so continue with search
        engine = get_search_engine()
        results, _ = engine.search(ocr_text)
        
        # Get top 10 results
        top_results = results[:10]
        
        # Get all documents for display
        documents = Document.objects.all().order_by('id')
        
        threading.Thread(target=engine.run_evaluation_async, daemon=True).start()
        
        return render(request, 'index.html', {
            'documents': documents,
            'results': top_results,
            'ocr_text': ocr_text,
            'search_performed': True
        })

    return redirect('index')

@csrf_exempt
def add_document(request):
    """Add a new document"""
    if request.method == 'POST' and request.FILES.get('document'):
        document_file = request.FILES['document']
        
        # Check if file is PDF
        if not document_file.name.lower().endswith('.pdf'):
            messages.error(request, 'Hanya file PDF yang diperbolehkan')
            return redirect('index')
        
        # Check file size (max 200KB)
        if document_file.size > MAX_UPLOAD_SIZE:
            messages.error(request, 'File Lebih dari 200 kb')
            return redirect('index')
        
        # Create a temporary file to check for duplicate content
        temp_dir = os.path.join(settings.BASE_DIR, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, document_file.name)
        
        with open(temp_path, 'wb+') as destination:
            for chunk in document_file.chunks():
                destination.write(chunk)
        
        # Calculate file hash for content comparison
        file_hash = calculate_file_hash(temp_path)
        
        # Check for duplicate title
        document_name = os.path.splitext(document_file.name)[0]
        if Document.objects.filter(document_name=document_name).exists():
            # Clean up temporary file
            os.remove(temp_path)
            messages.error(request, 'Dokumen sudah ada didalam database')
            return redirect('index')
        
        # Check for duplicate content by comparing file hash
        duplicate_content = False
        for doc in Document.objects.all():
            doc_path = doc.document_path
            if not os.path.isabs(doc_path):
                doc_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(doc_path))
            
            if os.path.exists(doc_path):
                existing_hash = calculate_file_hash(doc_path)
                if existing_hash == file_hash:
                    duplicate_content = True
                    break
        
        if duplicate_content:
            # Clean up temporary file
            os.remove(temp_path)
            messages.error(request, 'Dokumen sudah ada didalam database')
            return redirect('index')
        
        # Save file to disk
        pdf_storage_path = settings.PDF_STORAGE_PATH
        os.makedirs(pdf_storage_path, exist_ok=True)
        
        # Create a safe filename
        filename = document_file.name
        file_path = os.path.join(pdf_storage_path, filename)
        
        # Move the temporary file to the final location
        shutil.move(temp_path, file_path)
        
        # Create relative path for database
        relative_path = os.path.join('static/pdfdocuments', filename)
        
        # Add to database with transaction to ensure ID is updated correctly
        with transaction.atomic():
            # Get the highest ID
            max_id = Document.objects.all().order_by('-id').first()
            new_id = 1 if max_id is None else max_id.id + 1
            
            # Create new document record
            document = Document(
                id=new_id,
                document_name=document_name,
                document_uploaddate=datetime.date.today(),
                document_path=relative_path
            )
            document.save()
            
            # Reorder IDs to ensure they are sequential
            reorder_document_ids()
        
        # Rebuild the index since we added a document
        engine = get_search_engine()
        engine.build_index(force=True)
        
        messages.success(request, 'Dokumen berhasil ditambahkan')
        return redirect('index')

    return redirect('index')

def calculate_file_hash(file_path):
    """Calculate MD5 hash of a file for content comparison"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

@csrf_exempt
def delete_document(request, document_id):
    """Delete a document"""
    try:
        with transaction.atomic():
            # Get the document
            document = Document.objects.get(id=document_id)
            
            # Delete the file
            file_path = document.document_path
            if not os.path.isabs(file_path):
                file_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(file_path))
            
            if os.path.exists(file_path):
                os.remove(file_path)
            
            engine = get_search_engine()
            engine.document_processor.clear_document_cache(file_path)
            engine.word2vec_model.clear_document_vector(document_id)
            engine.transformer.clear_document_embedding(document_id)
            
            # Delete from database
            document.delete()
            
            # Reorder IDs to ensure they are sequential
            reorder_document_ids()
        
        # Rebuild the index since we deleted a document
        engine = get_search_engine()
        engine.build_index(force=True)
        
        messages.success(request, 'Dokumen berhasil dihapus')
        return redirect('index')
    except Document.DoesNotExist:
        messages.error(request, 'Dokumen tidak ditemukan')
        return redirect('index')
    except Exception as e:
        messages.error(request, f'Error: {str(e)}')
        return redirect('index')

def download_document(request, document_id):
    """Download a document"""
    try:
        document = Document.objects.get(id=document_id)
        file_path = document.document_path
        
        if not os.path.isabs(file_path):
            file_path = os.path.join(settings.PDF_STORAGE_PATH, os.path.basename(file_path))
        
        if os.path.exists(file_path):
            return FileResponse(open(file_path, 'rb'), as_attachment=True, filename=os.path.basename(file_path))
        else:
            messages.error(request, 'File tidak ditemukan')
            return redirect('index')
    except Document.DoesNotExist:
        messages.error(request, 'Dokumen tidak ditemukan')
        return redirect('index')
    except Exception as e:
        messages.error(request, str(e))
        return redirect('index')

def reorder_document_ids():
    """Reorder document IDs to ensure they are sequential"""
    documents = Document.objects.all().order_by('id')
    
    # Update IDs to be sequential
    for i, doc in enumerate(documents, start=1):
        if doc.id != i:
            Document.objects.filter(id=doc.id).update(id=i)
