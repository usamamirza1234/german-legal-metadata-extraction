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
        Extract text using improved OCR with multiple attempts.
        Saves all intermediate files under 'output_dir/{pdf_name}/'.
        """
        try:
            if self.debug:
                print("✅ ExtractorOCRText.extract_with_improved_ocr: ")
                print(f"      🖼️ Converting page {page_num} to high-quality image...")

            # Convert PDF page to image with higher DPI
            pages = convert_from_path(pdf_path, first_page=page_num, last_page=page_num, dpi=300)

            if not pages:
                return ""

            # Get base PDF name without extension
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

            # Define structured output directory
            output_dir = os.path.join("output_dir", pdf_name)
            os.makedirs(output_dir, exist_ok=True)

            # Save the original page image
            image_path = os.path.join(output_dir, f"page_{page_num}.png")
            original_image = pages[0]
            original_image.save(image_path)

            if self.debug:
                print(f"💾 Saved original image as {image_path}")

            # Initialize preprocessor
            preprocessor = ImagePreprocessor(image_path)
            save_intermediate = True

            # 1. Invert image for better OCR results
            inverted_image = preprocessor.invert_image()
            if save_intermediate:
                inverted_path = os.path.join(output_dir, f"inverted_page_{page_num}.jpg")
                cv2.imwrite(inverted_path, inverted_image)

                rescale_image = preprocessor.rescale_image(inverted_image,  2.5)
                rescale_image_path = os.path.join(output_dir, f"rescale_image_page_{page_num}.jpg")
                cv2.imwrite(rescale_image_path, rescale_image)

                # binarize_image = preprocessor.binarize_image(rescale_image, threshold= 115)
                # binarize_image_path = os.path.join(output_dir, f"binarize_image_page_{page_num}.jpg")
                # cv2.imwrite(binarize_image_path, binarize_image)


                # grayscale_image = preprocessor.grayscale(rescale_image)
                # grayscale_image_path = os.path.join(output_dir, f"grayscale_image_page_{page_num}.jpg")
                # cv2.imwrite(grayscale_image_path, grayscale_image)


                if self.debug:
                    print("Text after inversion:")

                # Return extracted text
                return preprocessor.extract_fraktur_text(rescale_image_path)

        except Exception as e:
            print(f"      ❌ OCR extraction failed: {e}")
            return ""

