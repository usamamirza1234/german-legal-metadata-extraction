"""
PDF Page Processor for OCR and Metadata Extraction
Integrates with existing ImagePreprocessor and ExtractorMetadata classes
"""

import cv2
import os
import fitz  # PyMuPDF
from typing import List, Dict


class PDFPageProcessor:
    """
    Process PDF pages for OCR and metadata extraction.
    Converts PDF pages to images, applies preprocessing, and extracts text/metadata.
    """

    def __init__(self, pdf_path: str, output_dir: str = "temp_pdf_processing", debug: bool = False):
        """
        Initialize PDF processor.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save intermediate files
            debug: Enable debug output
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.debug = debug

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Open PDF
        try:
            self.pdf_document = fitz.open(pdf_path)
            self.total_pages = len(self.pdf_document)
            if self.debug:
                print(f"📄 Loaded PDF with {self.total_pages} pages")
        except Exception as e:
            raise FileNotFoundError(f"Could not open PDF file '{pdf_path}': {e}")

    def convert_page_to_image(self, page_num: int, dpi: int = 300) -> str:
        """
        Convert a specific PDF page to image.

        Args:
            page_num: Page number (0-based)
            dpi: Resolution for conversion

        Returns:
            Path to saved image file
        """
        if page_num >= self.total_pages:
            raise ValueError(f"Page {page_num} does not exist. PDF has {self.total_pages} pages.")

        # Get page
        page = self.pdf_document[page_num]

        # Convert to image
        mat = fitz.Matrix(dpi / 72, dpi / 72)  # 72 is default DPI
        pix = page.get_pixmap(matrix=mat)

        # Save image
        image_path = os.path.join(self.output_dir, f"page_{page_num:03d}.png")
        pix.save(image_path)

        if self.debug:
            print(f"📸 Converted page {page_num} to {image_path}")

        return image_path

    def process_single_page(self, page_num: int, preprocessor_class, extractor_class,
                            preprocessing_steps: List[str] = None,
                            save_intermediate: bool = True) -> Dict:
        """
        Process a single PDF page through OCR pipeline.

        Args:
            page_num: Page number (0-based)
            preprocessor_class: ImagePreprocessor class
            extractor_class: ExtractorMetadata class
            preprocessing_steps: List of preprocessing steps to apply
            save_intermediate: Save intermediate processing steps

        Returns:
            Dictionary with extracted text and metadata
        """
        if self.debug:
            print(f"\n🔄 Processing page {page_num + 1}/{self.total_pages}")

        # Convert page to image
        image_path = self.convert_page_to_image(page_num)

        # Create page-specific output directory
        page_output_dir = os.path.join(self.output_dir, f"page_{page_num:03d}")
        os.makedirs(page_output_dir, exist_ok=True)

        # Initialize preprocessor
        preprocessor = preprocessor_class(image_path)

        # Apply preprocessing pipeline
        if preprocessing_steps is None:
            # Default preprocessing steps
            processed_image = preprocessor.preprocess_pipeline(
                output_dir=f"{page_output_dir}/",
                save_intermediate=save_intermediate
            )
            final_image_path = os.path.join(page_output_dir, "image_with_border.jpg")
        else:
            # Custom preprocessing steps
            processed_image = self._apply_custom_preprocessing(
                preprocessor, preprocessing_steps, page_output_dir, save_intermediate
            )
            final_image_path = os.path.join(page_output_dir, "final_processed.jpg")
            cv2.imwrite(final_image_path, processed_image)

        # Extract text using Fraktur OCR
        try:
            extracted_text = preprocessor.extract_fraktur_text(final_image_path)
            if self.debug:
                print(f"📝 Extracted text length: {len(extracted_text)} characters")
        except Exception as e:
            if self.debug:
                print(f"❌ OCR failed: {e}")
            extracted_text = ""

        # Extract metadata
        extractor = extractor_class(debug=self.debug)
        extracted_date = extractor.extract_date(extracted_text)

        # Prepare results
        result = {
            'page_number': page_num,
            'image_path': image_path,
            'processed_image_path': final_image_path,
            'extracted_text': extracted_text,
            'extracted_date': extracted_date,
            'text_length': len(extracted_text),
            'processing_directory': page_output_dir
        }

        if self.debug:
            print(f"✅ Page {page_num} processed successfully")
            if extracted_date:
                print(f"📅 Found date: {extracted_date}")

        return result

    def process_page_range(self, start_page: int, end_page: int,
                           preprocessor_class, extractor_class,
                           preprocessing_steps: List[str] = None,
                           save_intermediate: bool = True) -> List[Dict]:
        """
        Process a range of PDF pages.

        Args:
            start_page: Starting page number (0-based)
            end_page: Ending page number (0-based, inclusive)
            preprocessor_class: ImagePreprocessor class
            extractor_class: ExtractorMetadata class
            preprocessing_steps: List of preprocessing steps to apply
            save_intermediate: Save intermediate processing steps

        Returns:
            List of dictionaries with results for each page
        """
        if start_page < 0 or end_page >= self.total_pages:
            raise ValueError(f"Invalid page range. PDF has pages 0-{self.total_pages - 1}")

        if start_page > end_page:
            raise ValueError("Start page must be <= end page")

        results = []

        for page_num in range(start_page, end_page + 1):
            try:
                result = self.process_single_page(
                    page_num, preprocessor_class, extractor_class,
                    preprocessing_steps, save_intermediate
                )
                results.append(result)
            except Exception as e:
                if self.debug:
                    print(f"❌ Failed to process page {page_num}: {e}")
                # Add error result
                results.append({
                    'page_number': page_num,
                    'error': str(e),
                    'extracted_text': '',
                    'extracted_date': None,
                    'text_length': 0
                })

        return results

    def process_all_pages(self, preprocessor_class, extractor_class,
                          preprocessing_steps: List[str] = None,
                          save_intermediate: bool = True) -> List[Dict]:
        """
        Process all pages in the PDF.

        Args:
            preprocessor_class: ImagePreprocessor class
            extractor_class: ExtractorMetadata class
            preprocessing_steps: List of preprocessing steps to apply
            save_intermediate: Save intermediate processing steps

        Returns:
            List of dictionaries with results for each page
        """
        return self.process_page_range(
            0, self.total_pages - 1, preprocessor_class, extractor_class,
            preprocessing_steps, save_intermediate
        )

    def _apply_custom_preprocessing(self, preprocessor, steps: List[str],
                                    output_dir: str, save_intermediate: bool):
        """
        Apply custom preprocessing steps.

        Args:
            preprocessor: ImagePreprocessor instance
            steps: List of method names to apply
            output_dir: Directory to save intermediate files
            save_intermediate: Save intermediate files

        Returns:
            Final processed image
        """
        current_image = preprocessor.original_image

        for i, step in enumerate(steps):
            if self.debug:
                print(f"  Applying step {i + 1}: {step}")

            if hasattr(preprocessor, step):
                method = getattr(preprocessor, step)
                if step in ['rescale_image', 'binarize_image', 'add_borders']:
                    # Methods that might need parameters
                    current_image = method(current_image)
                else:
                    current_image = method(current_image)

                if save_intermediate:
                    step_path = os.path.join(output_dir, f"step_{i + 1:02d}_{step}.jpg")
                    cv2.imwrite(step_path, current_image)
            else:
                if self.debug:
                    print(f"  ⚠️ Warning: Method '{step}' not found")

        return current_image

    def get_summary_report(self, results: List[Dict]) -> Dict:
        """
        Generate a summary report of processing results.

        Args:
            results: List of processing results

        Returns:
            Summary dictionary
        """
        total_pages = len(results)
        successful_pages = len([r for r in results if 'error' not in r])
        failed_pages = total_pages - successful_pages

        dates_found = [r['extracted_date'] for r in results if r.get('extracted_date')]
        total_text_length = sum(r.get('text_length', 0) for r in results)

        summary = {
            'total_pages_processed': total_pages,
            'successful_pages': successful_pages,
            'failed_pages': failed_pages,
            'success_rate': f"{(successful_pages / total_pages * 100):.1f}%" if total_pages > 0 else "0%",
            'dates_found': len(dates_found),
            'unique_dates': list(set(dates_found)),
            'total_text_extracted': total_text_length,
            'average_text_per_page': total_text_length / successful_pages if successful_pages > 0 else 0
        }

        return summary

    def cleanup(self):
        """Clean up temporary files and close PDF."""
        if hasattr(self, 'pdf_document'):
            self.pdf_document.close()

        # Optionally remove temporary directory
        # shutil.rmtree(self.output_dir, ignore_errors=True)

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()

