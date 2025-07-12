import pdfplumber
import pytesseract
from pdf2image import convert_from_path
import os

from extractors.extractor_german_date import ExtractorGermanDate
from extractors.extractor_metadata import ExtractorMetadata
from extractors.extractor_ocr_text import ExtractorOCRText


class PDFExtractorWithOCR:
    def __init__(self, debug=False):
        self.debug = debug
        if self.debug:
            print("✅ PDF Extractor with OCR initialized")

        pytesseract.get_tesseract_version()
        print("✅ Tesseract OCR is available")



    #Step 1,
    def extract_text_from_pdf(self, pdf_path, start_page=1, end_page=None):
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
            text = extractor.extract_text_from_pdf(pdf_path, start_page=start_page, end_page=end_page)
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
        metadata_extractors = ExtractorMetadata(self.debug)
        metadata['date'] = metadata_extractors.extract_date(text)
        return metadata


    # def extract_metadata(self, text):
    #     """Extract metadata using pattern matching"""
    #     if not text.strip():
    #         return {"error": "No text to process"}
    #
    #     metadata = {}
    #
    #     # Extract year - look for 4-digit years
    #     years = re.findall(self.year_pattern, text)
    #     if years:
    #         full_years = [int(''.join(year)) for year in years]
    #         reasonable_years = [y for y in full_years if 1920 <= y <= 2025]
    #         if reasonable_years:
    #             metadata['year'] = reasonable_years[-1]  # Take the last reasonable year
    #
    #     # Extract title - look for substantial lines
    #     lines = [line.strip() for line in text.split('\n') if line.strip()]
    #     potential_titles = []
    #
    #     for line in lines[:20]:  # Check first 20 lines
    #         # Filter out page numbers, dates, and short lines
    #         if (len(line) > 20 and
    #                 not re.match(r'^[\d\s\-/.,:]+$', line) and
    #                 not re.search(r'Seite\s*\d+', line, re.IGNORECASE) and
    #                 not re.search(r'Page\s*\d+', line, re.IGNORECASE)):
    #             potential_titles.append(line)
    #
    #     if potential_titles:
    #         metadata['title'] = potential_titles[0]
    #
    #     # Extract German institutions
    #     institution_patterns = [
    #         r'(Reichsministerium[^.\n]+)',
    #         r'(Bundesministerium[^.\n]+)',
    #         r'(Ministerium[^.\n]+)',
    #         r'(Reichsamt[^.\n]+)',
    #         r'(Bundesamt[^.\n]+)',
    #         r'(Reichsregierung)',
    #         r'(Bundesregierung)',
    #     ]
    #
    #     for pattern in institution_patterns:
    #         matches = re.findall(pattern, text, re.IGNORECASE)
    #         if matches:
    #             metadata['publisher'] = matches[0]
    #             break
    #
    #     # Extract document type
    #     doc_types = {
    #         'Verordnung': r'\bVerordnung\b',
    #         'Gesetz': r'\bGesetz\b',
    #         'Beschluss': r'\bBeschluss\b',
    #         'Anordnung': r'\bAnordnung\b',
    #         'Prüfungsordnung': r'\bPrüfungsordnung\b',
    #         'Prüfungsanforderungen': r'\bPrüfungsanforderungen\b'
    #     }
    #
    #     for doc_type, pattern in doc_types.items():
    #         if re.search(pattern, text[:1000], re.IGNORECASE):
    #             metadata['document_type'] = doc_type
    #             break
    #
    #     return metadata
    #
    # def manual_label_document(self, pdf_path):
    #     """Manual labeling interface"""
    #     print(f"\n{'=' * 60}")
    #     print(f"📝 MANUAL LABELING: {os.path.basename(pdf_path)}")
    #     print(f"{'=' * 60}")
    #
    #     # Extract text with OCR
    #     text = self.extract_text_from_pdf(pdf_path)
    #
    #     if not text.strip():
    #         print("❌ Could not extract any text from this PDF!")
    #         return None
    #
    #     print(f"\n📖 Extracted text preview (first 800 characters):")
    #     print("-" * 60)
    #     print(text[:800])
    #     print("-" * 60)
    #
    #     # Manual input
    #     print(f"\n📋 Please enter metadata for this document:")
    #     title = input("🔤 Title: ").strip()
    #     year = input("📅 Year: ").strip()
    #     publisher = input("🏢 Publisher: ").strip()
    #     author = input("👤 Author: ").strip()
    #     doc_type = input("📄 Document Type: ").strip()
    #
    #     # Save labeled data
    #     label_data = {
    #         'filename': pdf_path,
    #         'text': text,
    #         'metadata': {
    #             'title': title,
    #             'year': int(year) if year.isdigit() else year,
    #             'publisher': publisher,
    #             'author': author,
    #             'document_type': doc_type
    #         }
    #     }
    #
    #     label_file = f"label_{os.path.basename(pdf_path).replace('.pdf', '.json')}"
    #     with open(label_file, 'w', encoding='utf-8') as f:
    #         json.dump(label_data, f, ensure_ascii=False, indent=2)
    #
    #     print(f"✅ Labeled data saved to {label_file}")
    #     return label_data
