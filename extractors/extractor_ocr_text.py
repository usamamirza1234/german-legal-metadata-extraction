"""
Improved OCR Extraction for Historical German Documents
Handles Fraktur script and poor image quality
"""

import os

import cv2
import pytesseract
from pdf2image import convert_from_path  # converts each PDF page to an image

from processor.ocr_image_processing import ImagePreprocessor


class ExtractorOCRText:
    def __init__(self, debug=False):
        self.debug = debug

    def extract_text_from_pdf(self, pdf_path, start_page=1, end_page=None):
        """
        Extract text from PDF with improved OCR
        """
        text = ""
        if self.debug:
            print("✅ ExtractorOCRText.extract_text_from_pdf: ")
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if end_page is None or end_page > total_pages:
                    end_page = total_pages

                if self.debug:
                    print(f"📄 Processing pages {start_page} to {end_page} of {total_pages} ..... "
                          f"in {os.path.basename(pdf_path)}")

                for i in range(start_page - 1, end_page):
                    page = pdf.pages[i]
                    print(f"📖 Processing page {i + 1}...")

                    ocr_text = self.extract_with_improved_ocr(pdf_path, page_num=i + 1)

                    if ocr_text and ocr_text.strip():
                        print(f"   ✅ OCR extraction: {len(ocr_text)} characters")
                        text += f"\n--- Page {i + 1} (OCR) ---\n{ocr_text}"
                    else:
                        print(f"   ❌ No text found on page {i + 1}")


        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            return ""

        return text

    def extract_with_improved_ocr(self, pdf_path, page_num=1):
        """
        Extract text using improved OCR with multiple attempts
        """
        try:
            if self.debug:
                print("✅ ExtractorOCRText.extract_with_improved_ocr: ")
                print(f"      🖼️ Converting page {page_num} to high-quality image...")

            # Convert with higher DPI for better quality
            pages = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300)
            output_dir = "output_dir/"
            os.makedirs(output_dir, exist_ok=True)

            if not pages:
                return ""

            os.path.basename(pdf_path)
            image_path = f"output_dir/{os.path.basename(pdf_path)}.png"
            original_image = pages[0]
            original_image.save(image_path)

            if self.debug:
                print(f"💾 Saved original image as ", image_path)

            # Initialize preprocessor
            preprocessor = ImagePreprocessor(image_path)
            save_intermediate = True

            # 1. Inverted Images
            inverted_image = preprocessor.invert_image()
            if save_intermediate:
                inverted_path = f"{output_dir}inverted.jpg"
                cv2.imwrite(inverted_path, inverted_image)
                print("Text after inversion:")
                # print(preprocessor.extract_fraktur_text(inverted_path))
                #preprocessor.display(inverted_path)
                return preprocessor.extract_fraktur_text(inverted_path)
        except Exception as e:
            print(f"      ❌ OCR extraction failed: {e}")
            return ""
