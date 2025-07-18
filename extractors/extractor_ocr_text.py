"""
Improved OCR Extraction for Historical German Documents
Handles Fraktur script and poor image quality
Now includes white space removal for better content focus
"""

import os

import cv2
import pytesseract
from pdf2image import convert_from_path  # converts each PDF page to an image

from processor.ocr_image_processing import ImagePreprocessor


class ExtractorOCRText:
    def __init__(self, debug=False, remove_white_spaces=True):
        """
        Initialize the OCR extractor

        Args:
            debug: Enable debug output
            remove_white_spaces: Boolean to control white space removal before processing
        """
        self.debug = debug
        self.remove_white_spaces = remove_white_spaces

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
        Now includes optional white space removal.
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

            # Start processing chain
            current_image = preprocessor.original_image

            # 0. Remove white areas FIRST (if enabled)
            if self.remove_white_spaces:
                if self.debug:
                    print("      🔍 Removing white areas...")

                # Try both methods and choose the better one
                method1_result = preprocessor.remove_white_areas(current_image)
                method2_result = preprocessor.smart_crop_content(current_image)

                # Save both results for comparison
                if save_intermediate:
                    method1_path = os.path.join(output_dir, f"white_removed_method1_page_{page_num}.jpg")
                    method2_path = os.path.join(output_dir, f"white_removed_method2_page_{page_num}.jpg")
                    cv2.imwrite(method1_path, method1_result)
                    cv2.imwrite(method2_path, method2_result)

                # Choose the method that results in more content (larger area)
                if method1_result.shape[0] * method1_result.shape[1] > method2_result.shape[0] * method2_result.shape[1]:
                    current_image = method1_result
                    chosen_method = "contour-based"
                else:
                    current_image = method2_result
                    chosen_method = "density-based"

                if self.debug:
                    print(f"      📏 Used {chosen_method} white removal method")

                # Save the chosen result
                if save_intermediate:
                    white_removed_path = os.path.join(output_dir, f"white_removed_page_{page_num}.jpg")
                    cv2.imwrite(white_removed_path, current_image)

            # 1. Invert image for better OCR results
            inverted_image = preprocessor.invert_image(current_image)
            if save_intermediate:
                inverted_path = os.path.join(output_dir, f"inverted_page_{page_num}.jpg")
                cv2.imwrite(inverted_path, inverted_image)

            # 2. Rescale image for better resolution
            rescale_image = preprocessor.rescale_image(inverted_image, 2.5)
            if save_intermediate:
                rescale_image_path = os.path.join(output_dir, f"rescale_image_page_{page_num}.jpg")
                cv2.imwrite(rescale_image_path, rescale_image)

            # Optional: Add binarization step (commented out but available)
            # binarize_image = preprocessor.binarize_image(rescale_image, threshold=115)
            # if save_intermediate:
            #     binarize_image_path = os.path.join(output_dir, f"binarize_image_page_{page_num}.jpg")
            #     cv2.imwrite(binarize_image_path, binarize_image)

            # Optional: Add grayscale conversion (commented out but available)
            # grayscale_image = preprocessor.grayscale(rescale_image)
            # if save_intermediate:
            #     grayscale_image_path = os.path.join(output_dir, f"grayscale_image_page_{page_num}.jpg")
            #     cv2.imwrite(grayscale_image_path, grayscale_image)

            if self.debug:
                print("      📝 Extracting text with Fraktur OCR...")

            # Extract text using the final processed image
            return preprocessor.extract_fraktur_text(rescale_image_path)

        except Exception as e:
            print(f"      ❌ OCR extraction failed: {e}")
            return ""

    def extract_with_custom_preprocessing(self, pdf_path, page_num=1, preprocessing_steps=None, end_page= None):
        """
        Extract text with custom preprocessing steps

        Args:
            pdf_path: Path to PDF file
            page_num: Page number to process
            preprocessing_steps: List of preprocessing steps to apply
                Available steps: ['remove_white', 'invert', 'rescale', 'grayscale', 'binarize', 'denoise']
        """
        if preprocessing_steps is None:
            preprocessing_steps = ['remove_white', 'invert', 'rescale']

        try:
            # Convert PDF page to image
            pages = convert_from_path(pdf_path, first_page=page_num, last_page=end_page, dpi=300)
            if not pages:
                return ""

            # Setup paths
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.join("output_dir", pdf_name)
            os.makedirs(output_dir, exist_ok=True)

            # Save original image
            image_path = os.path.join(output_dir, f"page_{page_num}.png")
            original_image = pages[0]
            original_image.save(image_path)

            # Initialize preprocessor
            preprocessor = ImagePreprocessor(image_path)
            current_image = preprocessor.original_image

            # Apply preprocessing steps in order
            for step in preprocessing_steps:
                if step == 'remove_white' and self.remove_white_spaces:
                    current_image = preprocessor.remove_white_areas(current_image)
                elif step == 'invert':
                    current_image = preprocessor.invert_image(current_image)
                elif step == 'rescale':
                    current_image = preprocessor.rescale_image(current_image, 2.5)
                elif step == 'grayscale':
                    current_image = preprocessor.grayscale(current_image)
                elif step == 'binarize':
                    current_image = preprocessor.binarize_image(current_image, threshold=115)
                elif step == 'denoise':
                    current_image = preprocessor.noise_removal(current_image)

                # Save intermediate result
                if self.debug:
                    step_path = os.path.join(output_dir, f"{step}_page_{page_num}.jpg")
                    cv2.imwrite(step_path, current_image)

            # Save final processed image
            final_path = os.path.join(output_dir, f"final_processed_page_{page_num}.jpg")
            cv2.imwrite(final_path, current_image)

            # Extract text from final processed image
            return preprocessor.extract_fraktur_text(final_path)

        except Exception as e:
            print(f"❌ Custom preprocessing failed: {e}")
            return ""

    def batch_process_with_different_methods(self, pdf_path, page_num=1):
        """
        Process the same page with different preprocessing combinations
        and return the best result based on text length
        """
        methods = [
            {
                'name': 'method_1_basic',
                'steps': ['remove_white', 'invert', 'rescale'],
                'remove_white': True
            },
            {
                'name': 'method_2_enhanced',
                'steps': ['remove_white', 'invert', 'rescale', 'binarize'],
                'remove_white': True
            },
            {
                'name': 'method_3_no_white_removal',
                'steps': ['invert', 'rescale'],
                'remove_white': False
            },
            {
                'name': 'method_4_grayscale',
                'steps': ['remove_white', 'grayscale', 'rescale'],
                'remove_white': True
            }
        ]

        results = {}

        for method in methods:
            try:
                # Temporarily set remove_white_spaces
                original_setting = self.remove_white_spaces
                self.remove_white_spaces = method['remove_white']

                if self.debug:
                    print(f"🔄 Testing {method['name']}...")

                text = self.extract_with_custom_preprocessing(
                    pdf_path,
                    page_num,
                    method['steps']
                )

                results[method['name']] = {
                    'text': text,
                    'length': len(text.strip()) if text else 0,
                    'steps': method['steps']
                }

                # Restore original setting
                self.remove_white_spaces = original_setting

            except Exception as e:
                print(f"❌ Method {method['name']} failed: {e}")
                results[method['name']] = {
                    'text': '',
                    'length': 0,
                    'steps': method['steps']
                }

        # Find the method with the best result (most text extracted)
        best_method = max(results.keys(), key=lambda k: results[k]['length'])

        if self.debug:
            print(f"📊 Results summary:")
            for method_name, result in results.items():
                print(f"   {method_name}: {result['length']} characters")
            print(f"🏆 Best method: {best_method}")

        return results[best_method]['text'], results

