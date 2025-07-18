import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import os

from extractors.extractor_german_date import ExtractorGermanDate
from extractors.extractor_metadata import ExtractedMetadata
from extractors.extractor_ocr_text import ExtractorOCRText


class PDFExtractorWithOCR:
    def __init__(self, debug=False):
        self.debug = debug

        if self.debug:
            print("✅ PDFExtractorWithOCR.init PDF Extractor with OCR initialized")

        pytesseract.get_tesseract_version()

        if self.debug:
            print("✅ Tesseract OCR is available")




    #Step 1,
    def extract_text_from_pdf(self, pdf_path, start_page=1, end_page=None, type=None, preprocessing_steps=None):
        """
        Extract text from a PDF between specified pages (inclusive).
        Falls back to OCR if direct extraction fails.

        Pages are 1-indexed (first page is 1).
        """
        text = ""

        try:
            if self.debug:
                print("✅ PDFExtractorWithOCR.extract_text_from_pdf: ")
            extractor = ExtractorOCRText(self.debug)


            if type is None:
                text = extractor.extract_text_from_pdf(pdf_path, start_page=start_page, end_page=end_page)
            elif type == "custom_preprocessing":
                text = extractor.extract_with_custom_preprocessing(pdf_path, page_num=start_page, end_page=end_page, preprocessing_steps=preprocessing_steps)
            # elif type == "batch_process":
            #     text, all_results = extractor.batch_process_with_different_methods(pdf_path, page_num=start_page, end_page=end_page)




        except Exception as e:
            print(f"❌ Error processing {pdf_path}: {e}")
            return ""

        return text



    #Step 2,
    def extract_metadata(self, text):
        """Extract metadata from text"""
        if not text.strip():
            return {"error": "No text to process"}

        if self.debug:
            print("✅ PDFExtractorWithOCR.extract_metadata: ")

        metadata = {}
        metadata_extractors = ExtractedMetadata(self.debug)
        return metadata_extractors.extract_date(text)


    def get_high_confidence_date(data):
        date_section = data.get('date', {})
        all_dates = date_section.get('all_dates', [])

        for date_entry in all_dates:
            if date_entry.get('confidence', '').lower() == 'high':
                return date_entry  # Return the first high confidence date found

        # If none found, fallback to the main date dict if its confidence is high
        if date_section.get('confidence', '').lower() == 'high':
            return date_section

        return None  # or {} if you prefer

