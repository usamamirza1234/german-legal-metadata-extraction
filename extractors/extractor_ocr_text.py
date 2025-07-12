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
            pages = convert_from_path(pdf_path, first_page=page_num, last_page=page_num,
                                      dpi=300)  # Higher DPI for better quality
            output_dir = "output_dir/"
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)


            if not pages:
                return ""
            image_path = "output_dir/original_page.png"
            original_image = pages[0]
            original_image.save(image_path)

            if self.debug:
                print("💾 Saved original image as 'output_dir/original_page.png'")

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
#
# def extract_with_improved_ocr(self, pdf_path, page_num=1):
#        """
#        Extract text using improved OCR with multiple attempts
#        """
#        try:
#            print(f"      🖼️ Converting page {page_num} to high-quality image...")
#
#            # Convert with higher DPI for better quality
#            pages = convert_from_path(pdf_path,
#                                      first_page=page_num,
#                                      last_page=page_num,
#                                      dpi=300)  # Higher DPI for better quality
#
#            if not pages:
#                return ""
#
#            original_image = pages[0]
#
#            # Save original for comparison
#            if self.debug:
#                original_image.save("temp/original_page.png")
#                print("      💾 Saved original image as 'temp/original_page.png'")
#
#            # Try multiple OCR approaches
#            ocr_results = []
#
#            # Approach 1: Original image with German
#            print(f"      🔤 OCR Attempt 1: Original image, German...")
#            try:
#                text1 = pytesseract.image_to_string(original_image, lang='deu')
#                ocr_results.append(("Original + German", text1))
#                if self.debug:
#                    print(f"         Result 1: {len(text1)} chars")
#            except Exception as e:
#                print(f"         Failed: {e}")
#
#            # Approach 2: Preprocessed image with German
#            print(f"      🔤 OCR Attempt 2: Preprocessed image, German...")
#            try:
#                preprocessed = self.preprocess_image(original_image)
#                text2 = pytesseract.image_to_string(preprocessed, lang='deu')
#                ocr_results.append(("Preprocessed + German", text2))
#                if self.debug:
#                    print(f"         Result 2: {len(text2)} chars")
#            except Exception as e:
#                print(f"         Failed: {e}")
#
#            # Approach 3: Try with Fraktur-specific settings
#            print(f"      🔤 OCR Attempt 3: Fraktur-optimized...")
#            try:
#                # Custom config for old German documents
#                custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzäöüßÄÖÜ0123456789().,- '
#                text3 = pytesseract.image_to_string(preprocessed, lang='deu', config=custom_config)
#                ocr_results.append(("Fraktur-optimized", text3))
#                if self.debug:
#                    print(f"         Result 3: {len(text3)} chars")
#            except Exception as e:
#                print(f"         Failed: {e}")
#
#            # Approach 4: Try with script-specific Tesseract model (if available)
#            print(f"      🔤 OCR Attempt 4: Script-specific...")
#            try:
#                # Try with frk (Fraktur) if available
#                text4 = pytesseract.image_to_string(preprocessed, lang='frk')
#                ocr_results.append(("Fraktur script", text4))
#                if self.debug:
#                    print(f"         Result 4: {len(text4)} chars")
#            except:
#                try:
#                    # Fallback to English on preprocessed
#                    text4 = pytesseract.image_to_string(preprocessed, lang='eng')
#                    ocr_results.append(("Preprocessed + English", text4))
#                    if self.debug:
#                        print(f"         Result 4 (English): {len(text4)} chars")
#                except Exception as e:
#                    print(f"         Failed: {e}")
#
#            # Choose the best result
#            if ocr_results:
#                # Prefer results with more characters and reasonable content
#                best_result = max(ocr_results, key=lambda x: len(x[1].strip()))
#
#                if self.debug:
#                    print(f"      🏆 Best result: {best_result[0]} with {len(best_result[1])} chars")
#                    print(f"      📖 Preview: {best_result[1][:200]}...")
#
#                return best_result[1]
#            else:
#                return ""
#
#        except Exception as e:
#            print(f"      ❌ OCR extraction failed: {e}")
#            return ""
