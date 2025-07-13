"""
Complete Integrated Metadata Extraction System
Thesis Implementation - Final Integration of All Components
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import asdict
import logging

from evaluation.metadata_evaluation import MetadataEvaluator
from extractors.enhanced_extractor_metadata import EnhancedMetadataExtractor
# Import all your existing modules
from extractors.extractor_german_date import ExtractorGermanDate
from extractors.extractor_meta_data_ml import MLMetadataExtractor, DatasetCreator
from extractors.extractor_metadata import ExtractedMetadata
from extractors.extractor_ocr_text import ExtractorOCRText
from pdf_extractor_with_ocr import PDFExtractorWithOCR

# Import the enhanced modules we created
# from enhanced_metadata_extractor import EnhancedMetadataExtractor, ExtractedMetadata
# from ml_training_pipeline import MLMetadataExtractor, DatasetCreator
# from evaluation_framework import MetadataEvaluator


class ComprehensiveExtractionSystem:
    """
    Complete system that integrates all components:
    1. OCR text extraction from PDFs
    2. Rule-based metadata extraction
    3. ML-based metadata extraction
    4. Ensemble methods
    5. Evaluation and reporting
    """

    def __init__(self, debug=False, use_ml=True):
        self.debug = debug
        self.use_ml = use_ml

        # Setup logging
        logging.basicConfig(level=logging.INFO if debug else logging.WARNING)
        self.logger = logging.getLogger(__name__)

        # Initialize all extractors
        self.pdf_extractor = PDFExtractorWithOCR(debug=debug)
        self.rule_based_extractor = EnhancedMetadataExtractor(debug=debug)

        if use_ml:
            self.ml_extractor = MLMetadataExtractor()
            self.models_loaded = False

        self.evaluator = MetadataEvaluator()

        # Results storage
        self.extraction_results = []
        self.performance_metrics = {}

    def load_ml_models(self, model_path: str):
        """Load pre-trained ML models"""
        if self.use_ml and os.path.exists(model_path):
            try:
                self.ml_extractor.load_models(model_path)
                self.models_loaded = True
                self.logger.info(f"✅ ML models loaded from {model_path}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to load ML models: {e}")
                self.models_loaded = False

    def train_ml_models(self, training_data_path: str, save_path: str = "trained_models.pkl"):
        """Train ML models on labeled data"""
        if not self.use_ml:
            return

        try:
            # Load training data
            if training_data_path.endswith('.json'):
                training_data = DatasetCreator.load_from_json(training_data_path)
            else:
                # Create sample data for testing
                training_data = DatasetCreator.create_sample_dataset()

            self.logger.info(f"📚 Training ML models on {len(training_data)} documents...")

            # Train models
            self.ml_extractor.train_all_models(training_data)

            # Save models
            self.ml_extractor.save_models(save_path)
            self.models_loaded = True

            self.logger.info(f"✅ ML models trained and saved to {save_path}")

        except Exception as e:
            self.logger.error(f"❌ ML training failed: {e}")
            self.models_loaded = False

    def extract_from_pdf(self, pdf_path: str, pages: Tuple[int, int] = None) -> Dict:
        """Extract metadata from a single PDF file"""

        self.logger.info(f"🔍 Processing: {os.path.basename(pdf_path)}")

        try:
            # Step 1: Extract text from PDF
            if pages:
                start_page, end_page = pages
                text = self.pdf_extractor.extract_text_from_pdf(pdf_path, start_page, end_page)
            else:
                text = self.pdf_extractor.extract_text_from_pdf(pdf_path)

            if not text.strip():
                return {
                    'filename': pdf_path,
                    'error': 'No text extracted from PDF',
                    'rule_based': None,
                    'ml_based': None,
                    'ensemble': None
                }

            # Step 2: Rule-based extraction
            rule_based_metadata = self.rule_based_extractor.extract_all_metadata(text)

            # Step 3: ML-based extraction (if available)
            ml_based_metadata = None
            if self.use_ml and self.models_loaded:
                try:
                    title, title_conf = self.ml_extractor.predict_title(text)
                    doc_type, type_conf = self.ml_extractor.predict_document_type(text)
                    publisher, pub_conf = self.ml_extractor.predict_publisher(text)

                    # Extract date using existing rule-based method (usually more reliable)
                    date_result = self.rule_based_extractor.german_date_extractor.find_publishing_date_with_details(
                        text)

                    ml_based_metadata = ExtractedMetadata(
                        title=title,
                        publisher=publisher,
                        document_type=doc_type,
                        year=date_result['date'].year if date_result else None,
                        date=date_result['date'] if date_result else None,
                        confidence_scores={
                            'title': title_conf,
                            'document_type': type_conf,
                            'publisher': pub_conf,
                            'date': 0.9 if date_result and date_result['confidence'] == 'high' else 0.6
                        }
                    )

                except Exception as e:
                    self.logger.warning(f"⚠️ ML extraction failed: {e}")
                    ml_based_metadata = None

            # Step 4: Ensemble method (combine rule-based and ML)
            ensemble_metadata = self._create_ensemble_result(rule_based_metadata, ml_based_metadata)

            # Package results
            result = {
                'filename': pdf_path,
                'text_length': len(text),
                'rule_based': asdict(rule_based_metadata) if rule_based_metadata else None,
                'ml_based': asdict(ml_based_metadata) if ml_based_metadata else None,
                'ensemble': asdict(ensemble_metadata) if ensemble_metadata else None,
                'extraction_timestamp': datetime.now().isoformat()
            }

            self.extraction_results.append(result)
            return result

        except Exception as e:
            self.logger.error(f"❌ Error processing {pdf_path}: {e}")
            return {
                'filename': pdf_path,
                'error': str(e),
                'rule_based': None,
                'ml_based': None,
                'ensemble': None
            }

    def _create_ensemble_result(self, rule_based: ExtractedMetadata,
                                ml_based: Optional[ExtractedMetadata]) -> ExtractedMetadata:
        """Create ensemble result by combining rule-based and ML predictions"""

        if not ml_based:
            return rule_based

        ensemble = ExtractedMetadata()
        ensemble_confidence = {}

        # Combine results field by field
        fields = ['title', 'publisher', 'document_type', 'author', 'year', 'date']

        for field in fields:
            rule_value = getattr(rule_based, field)
            ml_value = getattr(ml_based, field)

            rule_conf = rule_based.confidence_scores.get(field, 0.0) if rule_based.confidence_scores else 0.0
            ml_conf = ml_based.confidence_scores.get(field, 0.0) if ml_based.confidence_scores else 0.0

            # Choose the prediction with higher confidence
            if rule_conf >= ml_conf:
                setattr(ensemble, field, rule_value)
                ensemble_confidence[field] = rule_conf
                ensemble_confidence[f'{field}_source'] = 'rule_based'
            else:
                setattr(ensemble, field, ml_value)
                ensemble_confidence[field] = ml_conf
                ensemble_confidence[f'{field}_source'] = 'ml_based'

        ensemble.confidence_scores = ensemble_confidence
        return ensemble

    def batch_process_directory(self, directory_path: str, file_pattern: str = "*.pdf") -> List[Dict]:
        """Process all PDFs in a directory"""

        import glob

        pdf_files = glob.glob(os.path.join(directory_path, file_pattern))
        results = []

        self.logger.info(f"📁 Processing {len(pdf_files)} PDF files in {directory_path}")

        for i, pdf_path in enumerate(pdf_files, 1):
            self.logger.info(f"📄 Processing file {i}/{len(pdf_files)}: {os.path.basename(pdf_path)}")

            result = self.extract_from_pdf(pdf_path)
            results.append(result)

            # Progress update
            if i % 10 == 0:
                self.logger.info(f"✅ Completed {i}/{len(pdf_files)} files")

        return results

    def evaluate_performance(self, ground_truth_path: str) -> Dict:
        """Evaluate extraction performance against ground truth"""

        # Load ground truth
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)

        # Create lookup by filename
        gt_lookup = {os.path.basename(item['filename']): item['metadata'] for item in ground_truth}

        # Prepare predictions and ground truth for evaluation
        rule_predictions = []
        ml_predictions = []
        ensemble_predictions = []
        gt_data = []

        for result in self.extraction_results:
            filename = os.path.basename(result['filename'])

            if filename in gt_lookup:
                gt_data.append(gt_lookup[filename])

                # Rule-based predictions
                if result['rule_based']:
                    rule_predictions.append(result['rule_based'])
                else:
                    rule_predictions.append({})

                # ML predictions
                if result['ml_based']:
                    ml_predictions.append(result['ml_based'])
                else:
                    ml_predictions.append({})

                # Ensemble predictions
                if result['ensemble']:
                    ensemble_predictions.append(result['ensemble'])
                else:
                    ensemble_predictions.append({})

        # Evaluate each approach
        evaluation_results = {}

        if rule_predictions:
            evaluation_results['rule_based'] = self.evaluator.comprehensive_evaluation(rule_predictions, gt_data)

        if ml_predictions and any(ml_predictions):
            evaluation_results['ml_based'] = self.evaluator.comprehensive_evaluation(ml_predictions, gt_data)

        if ensemble_predictions:
            evaluation_results['ensemble'] = self.evaluator.comprehensive_evaluation(ensemble_predictions, gt_data)

        # Compare methods if we have both
        if 'rule_based' in evaluation_results and 'ml_based' in evaluation_results:
            evaluation_results['comparison'] = self.evaluator.compare_methods(
                rule_predictions, ml_predictions, gt_data
            )

        self.performance_metrics = evaluation_results
        return evaluation_results

    def generate_comprehensive_report(self, output_dir: str = "evaluation_results"):
        """Generate comprehensive evaluation report with visualizations"""

        os.makedirs(output_dir, exist_ok=True)

        # 1. Save raw results
        results_file = os.path.join(output_dir, "extraction_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.extraction_results, f, ensure_ascii=False, indent=2, default=str)

        # 2. Save performance metrics
        if self.performance_metrics:
            metrics_file = os.path.join(output_dir, "performance_metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.performance_metrics, f, ensure_ascii=False, indent=2, default=str)

        # 3. Generate text reports
        for approach in ['rule_based', 'ml_based', 'ensemble']:
            if approach in self.performance_metrics:
                report_file = os.path.join(output_dir, f"{approach}_report.txt")
                report = self.evaluator.generate_report(
                    self.performance_metrics[approach],
                    report_file
                )
                print(f"📊 {approach.replace('_', ' ').title()} report saved to {report_file}")

        # 4. Generate performance comparison visualization
        if 'comparison' in self.performance_metrics:
            plot_file = os.path.join(output_dir, "performance_comparison.png")
            self.evaluator.plot_performance_comparison(
                self.performance_metrics['comparison'],
                plot_file
            )
            print(f"📈 Performance comparison plot saved to {plot_file}")

        # 5. Generate summary statistics
        summary_file = os.path.join(output_dir, "summary_statistics.txt")
        self._generate_summary_statistics(summary_file)

        print(f"✅ Comprehensive report generated in {output_dir}")

    def _generate_summary_statistics(self, output_file: str):
        """Generate summary statistics"""

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("THESIS METADATA EXTRACTION - SUMMARY STATISTICS\n")
            f.write("=" * 60 + "\n\n")

            # Basic statistics
            f.write(f"Total documents processed: {len(self.extraction_results)}\n")

            successful_extractions = sum(1 for r in self.extraction_results if 'error' not in r)
            f.write(f"Successful extractions: {successful_extractions}\n")
            f.write(f"Success rate: {successful_extractions / len(self.extraction_results) * 100:.1f}%\n\n")

            # Field extraction rates
            if self.extraction_results:
                f.write("FIELD EXTRACTION RATES:\n")
                f.write("-" * 30 + "\n")

                fields = ['title', 'year', 'publisher', 'document_type', 'author']

                for approach in ['rule_based', 'ml_based', 'ensemble']:
                    f.write(f"\n{approach.replace('_', ' ').title()}:\n")

                    for field in fields:
                        extracted_count = 0
                        total_count = 0

                        for result in self.extraction_results:
                            if approach in result and result[approach]:
                                total_count += 1
                                if result[approach].get(field):
                                    extracted_count += 1

                        if total_count > 0:
                            rate = extracted_count / total_count * 100
                            f.write(f"  {field}: {extracted_count}/{total_count} ({rate:.1f}%)\n")

            # Performance summary
            if self.performance_metrics:
                f.write(f"\nPERFORMANCE SUMMARY:\n")
                f.write("-" * 30 + "\n")

                for approach in ['rule_based', 'ml_based', 'ensemble']:
                    if approach in self.performance_metrics:
                        overall = self.performance_metrics[approach].get('overall', {})
                        f1_score = overall.get('mean_f1_score', 0.0)
                        f.write(f"{approach.replace('_', ' ').title()} Mean F1-Score: {f1_score:.3f}\n")


# Main execution and example usage
def main():
    """Main function demonstrating the complete system"""

    # Initialize the complete system
    print("🚀 Initializing Comprehensive Metadata Extraction System...")
    system = ComprehensiveExtractionSystem(debug=True, use_ml=True)

    # Example 1: Train ML models (if you have training data)
    print("\n📚 Training ML models...")
    training_data_path = "training_data.json"  # Your labeled dataset

    if os.path.exists(training_data_path):
        system.train_ml_models(training_data_path)
    else:
        # Use sample data for demonstration
        print("⚠️ No training data found, using sample data for demonstration")
        system.train_ml_models("", "demo_models.pkl")

    # Example 2: Process individual PDF
    print("\n📄 Processing individual PDF...")
    pdf_path = "sample_document.pdf"  # Your PDF file

    if os.path.exists(pdf_path):
        result = system.extract_from_pdf(pdf_path)
        print(f"✅ Extraction completed for {pdf_path}")
        print(f"   Rule-based title: {result.get('rule_based', {}).get('title', 'Not found')}")
        print(f"   ML-based title: {result.get('ml_based', {}).get('title', 'Not found')}")
    else:
        print(f"⚠️ Sample PDF not found: {pdf_path}")

    # Example 3: Batch process directory
    print("\n📁 Batch processing directory...")
    pdf_directory = "documents/"  # Your directory with PDFs

    if os.path.exists(pdf_directory):
        results = system.batch_process_directory(pdf_directory)
        print(f"✅ Processed {len(results)} documents")
    else:
        print(f"⚠️ Document directory not found: {pdf_directory}")

    # Example 4: Evaluate performance (if you have ground truth)
    print("\n📊 Evaluating performance...")
    ground_truth_path = "ground_truth.json"  # Your ground truth labels

    if os.path.exists(ground_truth_path):
        evaluation = system.evaluate_performance(ground_truth_path)
        print("✅ Performance evaluation completed")
    else:
        print("⚠️ No ground truth data found for evaluation")

    # Example 5: Generate comprehensive report
    print("\n📋 Generating comprehensive report...")
    system.generate_comprehensive_report("thesis_results")
    print("✅ Comprehensive report generated")

    print("\n🎉 Thesis metadata extraction system demonstration complete!")


if __name__ == "__main__":
    main()