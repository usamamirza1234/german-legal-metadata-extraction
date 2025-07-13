from pdf_extractor_with_ocr import PDFExtractorWithOCR
from optimise_code.optimisation import OptimizedMetadataExtractor

extractor = OptimizedMetadataExtractor(debug=True)
pdf_extractor = PDFExtractorWithOCR(debug=False)
text = pdf_extractor.extract_text_from_pdf(pdf_path = "../pdfs/30s/30s/berufearchiv_5496.pdf", start_page=1, end_page=3)
print("EXTRACTION RESULTS: ", text)
# Extract metadata

metadata = extractor.extract_all_metadata(text)

print("\n" + "=" * 60)
print("EXTRACTION RESULTS:")
print("=" * 60)

result_dict = metadata.to_dict()
print(result_dict)
for key, value in result_dict.items():
    if value and key != 'raw_text_preview':
        print(f"{key.upper()}: {value}")
