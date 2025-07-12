from processor.pdf_page_processor import PDFPageProcessor
from processor.ocr_image_processing import ImagePreprocessor
from extractors.extractor_metadata import ExtractorMetadata

# Initialize PDF processor
pdf_processor = PDFPageProcessor(
    pdf_path="../pdfs/30s/30s/berufearchiv_5496.pdf",
    output_dir="temp/pdf_processing_output",
    debug=True
)

# Process specific pages (0-based indexing)
results = pdf_processor.process_page_range(
    start_page=0,
    end_page=0,  # Process first 5 pages
    preprocessor_class=ImagePreprocessor,
    extractor_class=ExtractorMetadata,
    save_intermediate=True
)

# Print results
for result in results:
    print(f"Page {result['page_number']}: Date={result['extracted_date']}")

# Get summary
summary = pdf_processor.get_summary_report(results)
print("\nSummary:", summary)

# Clean up
pdf_processor.cleanup()

# # Alternative: Process with custom preprocessing steps
# def example_custom_preprocessing():
#     """
#     Example with custom preprocessing steps.
#     """
#     from processor.ocr_image_processing import ImagePreprocessor
#     from extractors.extractor_metadata import ExtractorMetadata
#
#     pdf_processor = PDFPageProcessor("document.pdf", debug=True)
#
#     # Custom preprocessing pipeline
#     custom_steps = [
#         'grayscale',
#         'rescale_image',  # Will use default scale_factor
#         'binarize_image',  # Will use default threshold
#         'noise_removal',
#         'remove_borders',
#         'add_borders'
#     ]
#
#     results = pdf_processor.process_page_range(
#         start_page=0,
#         end_page=2,
#         preprocessor_class=ImagePreprocessor,
#         extractor_class=ExtractorMetadata,
#         preprocessing_steps=custom_steps,
#         save_intermediate=True
#     )
#
#     return results