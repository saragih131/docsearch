import os
import pytesseract
from PIL import Image
import tempfile
import logging
import sys
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flexible Tesseract configuration
if sys.platform.startswith('win'):
    # Try multiple possible paths on Windows
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        # Add tesseract to PATH environment variable instead of hardcoding
    ]
    
    tesseract_path = None
    for path in possible_paths:
        if os.path.exists(path):
            tesseract_path = path
            break
    
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        logger.info(f"Using Tesseract at: {tesseract_path}")
    else:
        logger.warning("Tesseract not found in common locations. Make sure it's installed and in your PATH.")
        # Don't set the path, let pytesseract try to find it in PATH
else:
    # On Linux/Mac, Tesseract should be in the PATH
    logger.info("Using system Tesseract installation")

class OCRProcessor:
    def __init__(self):
        # Check if Tesseract is available
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract is available (version: {version})")
            self.tesseract_available = True
        except Exception as e:
            logger.error(f"Tesseract is not available: {e}")
            logger.info("Please install Tesseract OCR:")
            if sys.platform.startswith('win'):
                logger.info("1. Download from: https://github.com/UB-Mannheim/tesseract/wiki")
                logger.info("2. Install and add to PATH environment variable")
            else:
                logger.info("1. Install using your package manager (e.g., apt-get install tesseract-ocr)")
                logger.info("2. Install language packs if needed (e.g., tesseract-ocr-ind)")
            self.tesseract_available = False
    
    def contains_text(self, image):
        """Check if the image contains any text"""
        if not self.tesseract_available:
            # If Tesseract is not available, assume there is text
            return True
            
        try:
            # Convert to grayscale for better text detection
            if image.mode != 'L':
                gray_image = image.convert('L')
            else:
                gray_image = image
                
            # Use Tesseract to get confidence data
            data = pytesseract.image_to_data(gray_image, output_type=pytesseract.Output.DICT)
            
            # Check if any text was detected with reasonable confidence
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            
            # If no confidences or average confidence is very low, probably no text
            if not confidences or (sum(confidences) / len(confidences) < 10):
                logger.info("No text detected in the image (low confidence)")
                return False
                
            # Check if any words were detected
            text = pytesseract.image_to_string(gray_image).strip()
            if not text:
                logger.info("No text detected in the image (empty string)")
                return False
                
            # If we have some text with reasonable confidence, there is text
            logger.info(f"Text detected in the image with avg confidence: {sum(confidences) / len(confidences):.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error checking for text: {e}")
            # If there's an error, assume there is text to be safe
            return True
    
    def process_image(self, image_file):
        """Process an image file and extract text using OCR"""
        try:
            # Open the image
            image = Image.open(image_file)
            
            # Check if the image contains text
            if not self.contains_text(image):
                return "GAMBAR TIDAK BERISI TEKS"
            
            # Extract text using pytesseract
            try:
                text = pytesseract.image_to_string(image, lang='ind+eng')  # Use Indonesian and English languages
            except Exception as e:
                logger.error(f"OCR processing error: {e}")
                logger.info("Falling back to default language")
                text = pytesseract.image_to_string(image)  # Fallback to default language
            
            if not text.strip():
                logger.warning("No text detected in the image")
                return "GAMBAR TIDAK BERISI TEKS"
            
            logger.info(f"OCR extracted {len(text)} characters")
            return text.strip()
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return "Error processing image"
    
    def process_image_from_path(self, image_path):
        """Process an image from a file path"""
        try:
            return self.process_image(open(image_path, 'rb'))
        except Exception as e:
            logger.error(f"Error opening image file: {e}")
            return "Error opening image file"
    
    def save_temp_image(self, image_file):
        """Save uploaded image to a temporary file and return the path"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                # Write the uploaded file to the temporary file
                for chunk in image_file.chunks():
                    temp_file.write(chunk)
                
                return temp_file.name
        except Exception as e:
            logger.error(f"Error saving temporary image: {e}")
            return None