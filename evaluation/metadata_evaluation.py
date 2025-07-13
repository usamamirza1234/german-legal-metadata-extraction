"""
Comprehensive Evaluation Framework for Metadata Extraction
Thesis Implementation - Performance Analysis and Benchmarking
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import re


class MetadataEvaluator:
    """
    Comprehensive evaluation framework for metadata extraction performance
    Includes metrics for both rule-based and ML approaches
    """

    def __init__(self):
        self.evaluation_results = {}
        self.detailed_results = []

    def evaluate_date_extraction(self, predictions: List[Optional[datetime]],
                                ground_truth: List[Optional[datetime]]) -> Dict[str, float]:
        """Evaluate date extraction performance"""

        # Convert to comparable format
        pred_years = [pred.year if pred else None for pred in predictions]
        true_years = [true.year if true else None for true in ground_truth]

        # Exact match accuracy
        exact_matches = sum(1 for p, t in zip(predictions, ground_truth) if p == t)
        exact_accuracy = exact_matches / len(predictions)

        # Year-only accuracy (more lenient)
        year_matches = sum(1 for p, t in zip(pred_years, true_years) if p == t)
        year_accuracy = year_matches / len(pred_years)

        # Year difference analysis (for partially correct dates)
        year_diffs = []
        for p, t in zip(pred_years, true_years):
            if p is not None and t is not None:
                year_diffs.append(abs(p - t))

        mean_year_error = np.mean(year_diffs) if year_diffs else float('inf')

        # Extraction rate (how many dates were found vs expected)
        found_rate = sum(1 for p in predictions if p is not None) / len(predictions)
        expected_rate = sum(1 for t in ground_truth if t is not None) / len(ground_truth)

        return {
            'exact_accuracy': exact_accuracy,
            'year_accuracy': year_accuracy,
            'mean_year_error': mean_year_error,
            'extraction_rate': found_rate,
            'expected_rate': expected_rate,
            'recall': found_rate / expected_rate if expected_rate > 0 else 0.0
        }

    def evaluate_text_extraction(self, predictions: List[Optional[str]],
                                ground_truth: List[Optional[str]],
                                field_name: str) -> Dict[str, float]:
        """Evaluate text field extraction (title, publisher, author)"""

        # Clean and normalize text for comparison
        def normalize_text(text):
            if not text:
                return ""
            # Remove extra whitespace, convert to lowercase
            return re.sub(r'\s+', ' ', text.strip().lower())

        pred_normalized = [normalize_text(p) for p in predictions]
        true_normalized = [normalize_text(t) for t in ground_truth]

        # Exact match accuracy
        exact_matches = sum(1 for p, t in zip(pred_normalized, true_normalized)
                           if p == t and p != "")
        total_with_truth = sum(1 for t in true_normalized if t != "")

        exact_accuracy = exact_matches / total_with_truth if total_with_truth > 0 else 0.0

        # Partial match accuracy (using word overlap)
        partial_matches = 0
        word_overlaps = []

        for p, t in zip(pred_normalized, true_normalized):
            if p and t:
                pred_words = set(p.split())
                true_words = set(t.split())

                if pred_words and true_words:
                    overlap = len(pred_words.intersection(true_words))
                    overlap_ratio = overlap / len(true_words)
                    word_overlaps.append(overlap_ratio)

                    if overlap_ratio >= 0.5:  # At least 50% word overlap
                        partial_matches += 1

        partial_accuracy = partial_matches / total_with_truth if total_with_truth > 0 else 0.0
        mean_word_overlap = np.mean(word_overlaps) if word_overlaps else 0.0

        # Extraction rate
        found_rate = sum(1 for p in predictions if p) / len(predictions)
        expected_rate = sum(1 for t in ground_truth if t) / len(ground_truth)

        # Precision and Recall
        true_positives = exact_matches
        false_positives = sum(1 for p, t in zip(pred_normalized, true_normalized)
                             if p != "" and (t == "" or p != t))
        false_negatives = sum(1 for p, t in zip(pred_normalized, true_normalized)
                             if t != "" and p == "")

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'exact_accuracy': exact_accuracy,
            'partial_accuracy': partial_accuracy,
            'mean_word_overlap': mean_word_overlap,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'extraction_rate': found_rate,
            'expected_rate': expected_rate
        }

    def evaluate_classification(self, predictions: List[str],
                              ground_truth: List[str],
                              field_name: str) -> Dict[str, float]:
        """Evaluate classification fields (document_type)"""

        # Handle None/empty values
        pred_clean = [p if p else "unknown" for p in predictions]
        true_clean = [t if t else "unknown" for t in ground_truth]

        # Basic metrics
        accuracy = accuracy_score(true_clean, pred_clean)
        precision, recall, f1, support = precision_recall_fscore_support(
            true_clean, pred_clean, average='weighted', zero_division=0
        )

        # Class-specific analysis
        unique_classes = list(set(true_clean + pred_clean))
        class_metrics = {}

        for cls in unique_classes:
            if cls == "unknown":
                continue

            cls_true = [1 if t == cls else 0 for t in true_clean]
            cls_pred = [1 if p == cls else 0 for p in pred_clean]

            if sum(cls_true) > 0:  # Only if class exists in ground truth
                cls_precision, cls_recall, cls_f1, _ = precision_recall_fscore_support(
                    cls_true, cls_pred, average='binary', zero_division=0
                )
                class_metrics[cls] = {
                    'precision': cls_precision,
                    'recall': cls_recall,
                    'f1_score': cls_f1,
                    'support': sum(cls_true)
                }

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'class_metrics': class_metrics
        }

    def comprehensive_evaluation(self, predictions: List[Dict],
                               ground_truth: List[Dict]) -> Dict[str, Dict]:
        """Perform comprehensive evaluation across all metadata fields"""

        results = {}

        # Extract individual fields
        fields = ['title', 'year', 'date', 'publisher', 'author', 'document_type']

        for field in fields:
            pred_values = [pred.get(field) for pred in predictions]
            true_values = [true.get(field) for true in ground_truth]

            if field == 'date':
                # Special handling for dates
                results[field] = self.evaluate_date_extraction(pred_values, true_values)
            elif field in ['title', 'publisher', 'author']:
                # Text fields
                results[field] = self.evaluate_text_extraction(pred_values, true_values, field)
            elif field in ['document_type']:
                # Classification fields
                results[field] = self.evaluate_classification(pred_values, true_values, field)
            elif field == 'year':
                # Year as integer
                # Convert dates to years if necessary
                pred_years = []
                true_years = []

                for p, t in zip(pred_values, true_values):
                    if isinstance(p, datetime):
                        pred_years.append(p.year)
                    else:
                        pred_years.append(p)

                    if isinstance(t, datetime):
                        true_years.append(t.year)
                    else:
                        true_years.append(t)

                # Treat as classification
                pred_years_str = [str(y) if y else None for y in pred_years]
                true_years_str = [str(y) if y else None for y in true_years]
                results[field] = self.evaluate_classification(pred_years_str, true_years_str, field)

        # Overall performance
        all_f1_scores = []
        for field, metrics in results.items():
            if 'f1_score' in metrics:
                all_f1_scores.append(metrics['f1_score'])

        results['overall'] = {
            'mean_f1_score': np.mean(all_f1_scores) if all_f1_scores else 0.0,
            'num_documents': len(predictions)
        }

        return results

    def compare_methods(self, rule_based_results: List[Dict],
                       ml_results: List[Dict],
                       ground_truth: List[Dict]) -> Dict:
        """Compare rule-based vs ML approaches"""

        rule_eval = self.comprehensive_evaluation(rule_based_results, ground_truth)
        ml_eval = self.comprehensive_evaluation(ml_results, ground_truth)

        comparison = {}

        for field in rule_eval:
            if field == 'overall':
                continue

            comparison[field] = {
                'rule_based': rule_eval[field],
                'ml_based': ml_eval[field],
                'improvement': {}
            }

            # Calculate improvement
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric in rule_eval[field] and metric in ml_eval[field]:
                    rule_score = rule_eval[field][metric]
                    ml_score = ml_eval[field][metric]
                    improvement = ml_score - rule_score
                    comparison[field]['improvement'][metric] = improvement

        return comparison

    def generate_error_analysis(self, predictions: List[Dict],
                               ground_truth: List[Dict],
                               texts: List[str]) -> Dict:
        """Generate detailed error analysis"""

        errors = {
            'missing_extractions': [],
            'incorrect_extractions': [],
            'partial_matches': []
        }

        for i, (pred, true, text) in enumerate(zip(predictions, ground_truth, texts)):
            doc_errors = {'doc_index': i, 'text_preview': text[:200] + '...'}

            for field in ['title', 'publisher', 'document_type']:
                pred_val = pred.get(field, '')
                true_val = true.get(field, '')

                if true_val and not pred_val:
                    # Missing extraction
                    doc_errors[f'missing_{field}'] = true_val
                elif pred_val and true_val and pred_val.lower() != true_val.lower():
                    # Incorrect extraction
                    doc_errors[f'incorrect_{field}'] = {
                        'predicted': pred_val,
                        'actual': true_val
                    }

            if any(key.startswith(('missing_', 'incorrect_')) for key in doc_errors.keys()):
                errors['incorrect_extractions'].append(doc_errors)

        return errors

    def plot_performance_comparison(self, comparison_results: Dict, save_path: str = None):
        """Create visualization of performance comparison"""

        fields = [f for f in comparison_results.keys() if f != 'overall']
        metrics = ['precision', 'recall', 'f1_score']

        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))

        for i, metric in enumerate(metrics):
            rule_scores = []
            ml_scores = []
            field_names = []

            for field in fields:
                if metric in comparison_results[field]['rule_based']:
                    rule_scores.append(comparison_results[field]['rule_based'][metric])
                    ml_scores.append(comparison_results[field]['ml_based'][metric])
                    field_names.append(field)

            x = np.arange(len(field_names))
            width = 0.35

            axes[i].bar(x - width/2, rule_scores, width, label='Rule-based', alpha=0.8)
            axes[i].bar(x + width/2, ml_scores, width, label='ML-based', alpha=0.8)

            axes[i].set_xlabel('Metadata Fields')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_xticks(x)
            axes[i].set_xticklabels(field_names, rotation=45)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def generate_report(self, results: Dict, output_file: str = None) -> str:
        """Generate comprehensive evaluation report"""

        report = []
        report.append("=" * 80)
        report.append("METADATA EXTRACTION EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Overall performance
        if 'overall' in results:
            report.append("OVERALL PERFORMANCE")
            report.append("-" * 40)
            report.append(f"Mean F1-Score: {results['overall']['mean_f1_score']:.3f}")
            report.append(f"Documents Evaluated: {results['overall']['num_documents']}")
            report.append("")

        # Field-by-field results
        for field, metrics in results.items():
            if field == 'overall':
                continue

            report.append(f"{field.upper()} EXTRACTION")
            report.append("-" * 40)

            for metric, value in metrics.items():
                if isinstance(value, dict):
                    if metric == 'class_metrics':
                        report.append(f"Class-specific metrics:")
                        for cls, cls_metrics in value.items():
                            report.append(f"  {cls}: F1={cls_metrics['f1_score']:.3f}, Support={cls_metrics['support']}")
                    continue
                elif isinstance(value, float):
                    report.append(f"{metric.replace('_', ' ').title()}: {value:.3f}")
                else:
                    report.append(f"{metric.replace('_', ' ').title()}: {value}")

            report.append("")

        report_text = "\n".join(report)

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)

        return report_text


# # Example usage and testing
# if __name__ == "__main__":
#     # Initialize evaluator
#     evaluator = MetadataEvaluator()
#
#     # Sample data for testing
#     ground_truth = [
#         {
#             'title': 'Verordnung über die Prüfungsanforderungen für Kaufleute',
#             'year': 1939,
#             'publisher': 'Reichsministerium für Wirtschaft',
#             'document_type': 'Prüfungsordnung',
#             'author': None
#         },
#         {
#             'title': 'Lehrplan für den Unterricht in der Berufsschule',
#             'year': 1925,
#             'publisher': 'Preußisches Ministerium für Wissenschaft',
#             'document_type': 'Lehrplan',
#             'author': None
#         }
#     ]
#
#     # ML predictions (generally better)
#     ml_predictions = [
#         {
#             'title': 'Verordnung über die Prüfungsanforderungen für Kaufleute',
#             'year': 1939,
#             'publisher': 'Reichsministerium für Wirtschaft',
#             'document_type': 'Prüfungsordnung',
#             'author': None
#         },
#         {
#             'title': 'Lehrplan für den Unterricht in der Berufsschule',
#             'year': 1925,
#             'publisher': 'Preußisches Ministerium für Wissenschaft',
#             'document_type': 'Lehrplan',
#             'author': None
#         }
#     ]
#
#     # Evaluate both approaches
#     print("EVALUATING RULE-BASED APPROACH:")
#     rule_results = evaluator.comprehensive_evaluation(rule_predictions, ground_truth)
#
#     print("\nEVALUATING ML-BASED APPROACH:")
#     ml_results = evaluator.comprehensive_evaluation(ml_predictions, ground_truth)
#
#     print("\nCOMPARING APPROACHES:")
#     comparison = evaluator.compare_methods(rule_predictions, ml_predictions, ground_truth)
#
#     # Generate report
#     print("\n" + "="*80)
#     print("RULE-BASED EVALUATION REPORT:")
#     print("="*80)
#     rule_report = evaluator.generate_report(rule_results)
#     print(rule_report)
#
#     print("\n" + "="*80)
#     print("ML-BASED EVALUATION REPORT:")
#     print("="*80)
#     ml_report = evaluator.generate_report(ml_results)
#     print(ml_report)
#
#     # Print comparison summary
#     print("\n" + "="*80)
#     print("COMPARISON SUMMARY:")
#     print("="*80)
#
#     for field, data in comparison.items():
#         if 'improvement' in data:
#             improvements = data['improvement']
#             if 'f1_score' in improvements:
#                 improvement = improvements['f1_score']
#                 print(f"{field.upper()}: F1-Score improvement = {improvement:+.3f}")
# ehrplan',
#             'author': None
#         }
#     ]
#
#     # Rule-based predictions (with some errors)
#     rule_predictions = [
#         {
#             'title': 'Verordnung über die Prüfungsanforderungen für Kaufleute',
#             'year': 1939,
#             'publisher': 'Reichsministerium für Wirtschaft',
#             'document_type': 'Verordnung',  # Slightly wrong
#             'author': None
#         },
#         {
#             'title': 'Lehrplan für Berufsschule',  # Partial match
#             'year': 1925,
#             'publisher': None,  # Missing
#             'document_type': 'L