# import pdfplumber
# import json
# import re
# import pytesseract
# from pdf2image import convert_from_path
# from PIL import Image
# import os
#
#
# class PDFExtractorWithOCR:
#     def __init__(self):
#         self.year_pattern = r'\b(19|20)\d{2}\b'
#         print("✅ PDF Extractor with OCR initialized")
#
#         # Test if Tesseract is available
#         try:
#             pytesseract.get_tesseract_version()
#             print("✅ Tesseract OCR is available")
#         except:
#             print("⚠️ Tesseract not found. Install from: https://github.com/UB-Mannheim/tesseract/wiki")
#             print("   For Windows: Download and install tesseract-ocr-w64-setup-v5.3.0.exe")
#
#     def extract_text_from_pdf(self, pdf_path, max_pages=3):
#         """Extract text from PDF, using OCR if needed"""
#         text = ""
#
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 print(f"📄 Processing {len(pdf.pages)} pages from {os.path.basename(pdf_path)}")
#
#                 for i, page in enumerate(pdf.pages[:max_pages]):
#                     print(f"📖 Processing page {i + 1}...")
#
#                     # Try to extract text directly first
#                     page_text = page.extract_text()
#
#                     if page_text and page_text.strip():
#                         print(f"   ✅ Direct text extraction: {len(page_text)} characters")
#                         text += f"\n--- Page {i + 1} ---\n{page_text}"
#                     else:
#                         print(f"   🔍 No direct text found, trying OCR...")
#                         # Use OCR as fallback
#                         ocr_text = self.extract_with_ocr(pdf_path, page_num=i + 1)
#                         if ocr_text:
#                             print(f"   ✅ OCR extraction: {len(ocr_text)} characters")
#                             text += f"\n--- Page {i + 1} (OCR) ---\n{ocr_text}"
#                         else:
#                             print(f"   ❌ No text found on page {i + 1}")
#
#         except Exception as e:
#             print(f"❌ Error processing {pdf_path}: {e}")
#             return ""
#
#         return text
#
#     def extract_with_ocr(self, pdf_path, page_num=1):
#         """Extract text using OCR from specific page"""
#         try:
#             # Convert PDF page to image
#             print(f"      🖼️ Converting page {page_num} to image...")
#             pages = convert_from_path(pdf_path,
#                                       first_page=page_num,
#                                       last_page=page_num,
#                                       dpi=200)  # Good quality for OCR
#
#             if not pages:
#                 return ""
#
#             page_image = pages[0]
#
#             # Try OCR with German language first, then English
#             try:
#                 print(f"      🔤 Running OCR (German)...")
#                 text = pytesseract.image_to_string(page_image, lang='deu')
#                 if text.strip():
#                     return text
#             except:
#                 print(f"      🔤 German OCR failed, trying English...")
#
#             # Fallback to English OCR
#             try:
#                 text = pytesseract.image_to_string(page_image, lang='eng')
#                 return text
#             except Exception as e:
#                 print(f"      ❌ OCR failed: {e}")
#                 return ""
#
#         except Exception as e:
#             print(f"      ❌ Image conversion failed: {e}")
#             return ""
#
#     def quick_pdf_analysis(self, pdf_path):
#         """Quick analysis to understand PDF structure"""
#         print(f"\n🔍 ANALYZING PDF: {os.path.basename(pdf_path)}")
#         print("=" * 60)
#
#         try:
#             with pdfplumber.open(pdf_path) as pdf:
#                 print(f"📊 Total pages: {len(pdf.pages)}")
#                 print(f"📊 PDF metadata: {pdf.metadata}")
#
#                 for i, page in enumerate(pdf.pages[:3]):
#                     print(f"\n📄 Page {i + 1} analysis:")
#
#                     # Check for direct text
#                     direct_text = page.extract_text()
#                     print(f"   📝 Direct text: {len(direct_text) if direct_text else 0} characters")
#
#                     # Check for images
#                     images = page.images
#                     print(f"   🖼️ Images found: {len(images)}")
#
#                     # Check page dimensions
#                     print(f"   📐 Page size: {page.width} x {page.height}")
#
#                     # Show sample of direct text if available
#                     if direct_text and direct_text.strip():
#                         print(f"   📖 Sample text: {direct_text[:100]}...")
#                     else:
#                         print(f"   ⚠️ No direct text - likely scanned image")
#
#         except Exception as e:
#             print(f"❌ Analysis failed: {e}")
#
#     def extract_metadata(self, text):
#         """Extract metadata using pattern matching"""
#         if not text.strip():
#             return {"error": "No text to process"}
#
#         metadata = {}
#
#         # Extract year - look for 4-digit years
#         years = re.findall(self.year_pattern, text)
#         if years:
#             full_years = [int(''.join(year)) for year in years]
#             reasonable_years = [y for y in full_years if 1920 <= y <= 2025]
#             if reasonable_years:
#                 metadata['year'] = reasonable_years[-1]  # Take the last reasonable year
#
#         # Extract title - look for substantial lines
#         lines = [line.strip() for line in text.split('\n') if line.strip()]
#         potential_titles = []
#
#         for line in lines[:20]:  # Check first 20 lines
#             # Filter out page numbers, dates, and short lines
#             if (len(line) > 20 and
#                     not re.match(r'^[\d\s\-/.,:]+$', line) and
#                     not re.search(r'Seite\s*\d+', line, re.IGNORECASE) and
#                     not re.search(r'Page\s*\d+', line, re.IGNORECASE)):
#                 potential_titles.append(line)
#
#         if potential_titles:
#             metadata['title'] = potential_titles[0]
#
#         # Extract German institutions
#         institution_patterns = [
#             r'(Reichsministerium[^.\n]+)',
#             r'(Bundesministerium[^.\n]+)',
#             r'(Ministerium[^.\n]+)',
#             r'(Reichsamt[^.\n]+)',
#             r'(Bundesamt[^.\n]+)',
#             r'(Reichsregierung)',
#             r'(Bundesregierung)',
#         ]
#
#         for pattern in institution_patterns:
#             matches = re.findall(pattern, text, re.IGNORECASE)
#             if matches:
#                 metadata['publisher'] = matches[0]
#                 break
#
#         # Extract document type
#         doc_types = {
#             'Verordnung': r'\bVerordnung\b',
#             'Gesetz': r'\bGesetz\b',
#             'Beschluss': r'\bBeschluss\b',
#             'Anordnung': r'\bAnordnung\b',
#             'Prüfungsordnung': r'\bPrüfungsordnung\b',
#             'Prüfungsanforderungen': r'\bPrüfungsanforderungen\b'
#         }
#
#         for doc_type, pattern in doc_types.items():
#             if re.search(pattern, text[:1000], re.IGNORECASE):
#                 metadata['document_type'] = doc_type
#                 break
#
#         return metadata
#
#     def manual_label_document(self, pdf_path):
#         """Manual labeling interface"""
#         print(f"\n{'=' * 60}")
#         print(f"📝 MANUAL LABELING: {os.path.basename(pdf_path)}")
#         print(f"{'=' * 60}")
#
#         # Extract text with OCR
#         text = self.extract_text_from_pdf(pdf_path)
#
#         if not text.strip():
#             print("❌ Could not extract any text from this PDF!")
#             return None
#
#         print(f"\n📖 Extracted text preview (first 800 characters):")
#         print("-" * 60)
#         print(text[:800])
#         print("-" * 60)
#
#         # Manual input
#         print(f"\n📋 Please enter metadata for this document:")
#         title = input("🔤 Title: ").strip()
#         year = input("📅 Year: ").strip()
#         publisher = input("🏢 Publisher: ").strip()
#         author = input("👤 Author: ").strip()
#         doc_type = input("📄 Document Type: ").strip()
#
#         # Save labeled data
#         label_data = {
#             'filename': pdf_path,
#             'text': text,
#             'metadata': {
#                 'title': title,
#                 'year': int(year) if year.isdigit() else year,
#                 'publisher': publisher,
#                 'author': author,
#                 'document_type': doc_type
#             }
#         }
#
#         label_file = f"label_{os.path.basename(pdf_path).replace('.pdf', '.json')}"
#         with open(label_file, 'w', encoding='utf-8') as f:
#             json.dump(label_data, f, ensure_ascii=False, indent=2)
#
#         print(f"✅ Labeled data saved to {label_file}")
#         return label_data
#
#
# def main():
#     extractor = PDFExtractorWithOCR()
#
#     # Get PDF file
#     # pdf_file = input("📁 Enter PDF path: ").strip()
#     # if not pdf_file:
#     #     pdf_file = "test.pdf"
#
#     pdf_file = "pdfs/30s/30s/30s_edelmetallpruefer_1938_pruefungsanforderungen.pdf"
#
#     if not os.path.exists(pdf_file):
#         print(f"❌ File not found: {pdf_file}")
#         return
#
#     # Step 1: Analyze PDF structure
#     extractor.quick_pdf_analysis(pdf_file)
#
#     # Step 2: Extract text
#     print(f"\n🚀 EXTRACTING TEXT...")
#     text = extractor.extract_text_from_pdf(pdf_file)
#
#     if text.strip():
#         print(f"\n✅ SUCCESS! Extracted {len(text)} characters total")
#
#         # Step 3: Auto-extract metadata
#         print(f"\n🤖 AUTOMATIC METADATA EXTRACTION...")
#         metadata = extractor.extract_metadata(text)
#         print("Results:")
#         for key, value in metadata.items():
#             print(f"   {key}: {value}")
#
#         # Step 4: Manual labeling option
#         choice = input(f"\n📝 Do manual labeling? (y/n): ").lower()
#         if choice == 'y':
#             extractor.manual_label_document(pdf_file)
#
#     else:
#         print(f"\n❌ Could not extract text. Possible issues:")
#         print("   • PDF is password protected")
#         print("   • Tesseract OCR is not installed properly")
#         print("   • PDF contains only images with unrecognizable text")
#
#
# if __name__ == "__main__":
#     main()