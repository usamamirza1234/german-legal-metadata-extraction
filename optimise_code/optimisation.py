"""
Optimized Metadata Extraction System for Historical German Documents
Consolidated and optimized version with improved performance
"""

import os
import re
import cv2
import json
import pickle
import numpy as np
import pytesseract
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from pdf2image import convert_from_path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


@dataclass
class ExtractedMetadata:
    """Structured metadata container with improved typing"""
    title: Optional[str] = None
    year: Optional[int] = None
    date: Optional[datetime] = None
    publisher: Optional[str] = None
    author: Optional[str] = None
    document_type: Optional[str] = None
    profession: Optional[str] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    raw_text_preview: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'title': self.title,
            'year': self.year,
            'date': self.date.isoformat() if self.date else None,
            'publisher': self.publisher,
            'author': self.author,
            'document_type': self.document_type,
            'profession': self.profession,
            'confidence_scores': self.confidence_scores,
            'raw_text_preview': self.raw_text_preview
        }


class OptimizedImageProcessor:
    """Optimized image processing for OCR with minimal overhead"""

    def __init__(self, debug: bool = False):
        self.debug = debug

    def process_for_ocr(self, image_path: str, output_dir: str = "output_dir/") -> str:
        """Streamlined processing pipeline for best OCR results"""
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Load and convert to grayscale in one step
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")

            # Apply optimized preprocessing
            # 1. Invert colors (often better for historical documents)
            inverted = cv2.bitwise_not(image)

            # 2. Apply adaptive thresholding for better results
            binary = cv2.adaptiveThreshold(
                inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )

            # 3. Minimal noise removal
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # Save processed image
            processed_path = os.path.join(output_dir, "processed_for_ocr.jpg")
            cv2.imwrite(processed_path, cleaned)

            if self.debug:
                print(f"✅ Image processed and saved to: {processed_path}")

            return processed_path

        except Exception as e:
            if self.debug:
                print(f"❌ Image processing failed: {e}")
            return image_path  # Return original path as fallback

    def extract_text_with_fraktur(self, image_path: str) -> str:
        """Extract text using German Fraktur model"""
        try:
            # Try Fraktur first, fallback to regular German
            text = pytesseract.image_to_string(
                cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),
                lang='deu_frak'
            )
            if not text.strip():
                text = pytesseract.image_to_string(
                    cv2.imread(image_path, cv2.IMREAD_GRAYSCALE),
                    lang='deu'
                )
            return text.strip()
        except Exception as e:
            if self.debug:
                print(f"❌ OCR extraction failed: {e}")
            return ""


class OptimizedMetadataExtractor:
    """Consolidated metadata extractor with all functionality"""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.image_processor = OptimizedImageProcessor(debug)
        self._init_patterns()

        # ML components
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}

    def _init_patterns(self):
        """Initialize all regex patterns in one place"""

        # German month mapping with OCR error corrections
        self.german_months = {
            'januar': 1, 'jan': 1, 'zamuar': 1, 'zanuar': 1, 'jamuar': 1, 'jänner': 1,
            'februar': 2, 'feb': 2,
            'märz': 3, 'mär': 3, 'maerz': 3,
            'april': 4, 'apr': 4, 'aprıl': 4,
            'mai': 5,
            'juni': 6, 'jun': 6,
            'juli': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'oktober': 10, 'okt': 10,
            'november': 11, 'nov': 11,
            'dezember': 12, 'dez': 12, 'deeember': 12, 'dezernber': 12
        }

        # Consolidated title patterns
        self.title_patterns = [
            r'(?:Berufs-?\s*)?([A-ZÄÖÜ][a-zäöüß]*anforderungen)\s*(?:\n.*?)?\s*(?:für|fü r)\s+([^.\n]{10,60})',
            r'(Berufs-?[Ee]ignungsanforderungen)\s*(?:\n.*?)?\s*(?:für.*?)\s*([A-ZÄÖÜ][a-zäöüß]{8,25})',
            r'(?:Verordnung|Anordnung|Gesetz|Bestimmungen|Richtlinien)\s+(?:über|für|zur|betreffend)\s+([^.\n]{20,100})',
            r'^([A-ZÄÖÜ][^.\n]{15,80})$'
        ]

        # Consolidated publisher patterns
        self.publisher_patterns = [
            r'bearbeitet\s+vom\s*\n?\s*([^\n.]{10,80})',
            r'(Deutscher\s+Ausschuss?\s+für\s+[^.\n]{5,40})\s*(?:\([^)]+\))?\s*[Ee]\.?[Vv]\.?',
            r'(Reichsministerium\s+für\s+[^.\n]+)',
            r'(Bundesministerium\s+für\s+[^.\n]+)',
            r'(Preußisches?\s+Ministerium\s+[^.\n]+)',
            r'(Deutsche\s+Arbeitsfront)',
            r'(Reichsgruppe\s+[A-ZÄÖÜ][a-zäöüß]+)',
            r'(Arbeitsgemeinschaft\s+[^.\n]{10,50})'
        ]

        # Document type patterns
        self.doc_type_patterns = {
            'Eignungsanforderungen': [
                r'[Ee]ignungsanforderungen', r'Berufs-?[Ee]ignungsanforderungen',
                r'Anforderungen.*Eintritt.*Lehrberuf'
            ],
            'Prüfungsordnung': [
                r'\bPrüfungsordnung\b', r'\bPrüfungsanforderungen\b',
                r'Ordnung.*Prüfung', r'Bestimmungen.*Prüfung'
            ],
            'Lehrplan': [
                r'\bLehrplan\b', r'\bLehrpläne\b', r'Plan.*Unterricht', r'Unterrichtsplan'
            ],
            'Ausbildungsordnung': [
                r'\bAusbildungsordnung\b', r'Ordnung.*Ausbildung', r'Bestimmungen.*Ausbildung'
            ],
            'Verordnung': [r'\bVerordnung\b', r'\bAnordnung\b'],
            'Gesetz': [r'\bGesetz\b', r'\bBerufsbildungsgesetz\b', r'\bHandwerksordnung\b']
        }

        # Profession patterns
        self.profession_patterns = [
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}fasser)',  # Steinfasser, etc.
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}macher)',  # Uhrmacher, etc.
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}schmidt)', # Goldschmidt, etc.
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}bauer)',   # Instrumentenbauer, etc.
            r'(Kaufmann|Mechaniker|Elektriker|Bäcker|Schneider|Tischler)'
        ]

        # Date publishing indicators (enhanced)
        self.publishing_indicators = [
            # Very strong indicators
            r'(?:Stand\s+vom|Stcmd\s+vom)',  # "Stand vom" or OCR error "Stcmd vom"
            r'(?:\(.*?Stand\s+vom.*?\))',     # "(Stand vom ...)" in parentheses
            r'(?:\(.*?Stcmd\s+vom.*?\))',     # "(Stcmd vom ...)" in parentheses

            # Strong indicators
            r'(?:Berlin,?\s+den)', r'(?:München,?\s+den)', r'(?:Hamburg,?\s+den)',
            r'(?:Datum\s*:)', r'(?:Ausgegeben\s+am)', r'(?:Verkündet\s+am)',

            # Medium indicators
            r'(?:Erlaß.*?vom)', r'(?:Erlass.*?vom)',  # "Erlaß ... vom"
            r'(?:mit\s+Wirkung\s+vom)',              # "mit Wirkung vom"
            r'(?:in\s+Kraft.*?vom)',                 # "in Kraft ... vom"
            r'(?:gültig\s+ab)',                      # "gültig ab"
        ]

        # Date pattern
        self.date_pattern = re.compile(r'(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})', re.IGNORECASE)

    def extract_text_from_pdf(self, pdf_path: str, start_page: int = 1, end_page: Optional[int] = None) -> str:
        """Extract text from PDF with optimized OCR"""
        if self.debug:
            print(f"📄 Processing PDF: {os.path.basename(pdf_path)}")

        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                if end_page is None or end_page > total_pages:
                    end_page = total_pages

                text = ""
                for i in range(start_page - 1, end_page):
                    if self.debug:
                        print(f"📖 Processing page {i + 1}...")

                    # Convert page to image for OCR
                    pages = convert_from_path(pdf_path, first_page=i+1, last_page=i+1, dpi=300)
                    if pages:
                        # Save temporary image
                        temp_path = f"temp_page_{i+1}.png"
                        pages[0].save(temp_path)

                        # Process and extract text
                        processed_path = self.image_processor.process_for_ocr(temp_path)
                        ocr_text = self.image_processor.extract_text_with_fraktur(processed_path)

                        if ocr_text.strip():
                            text += f"\n--- Page {i + 1} (OCR) ---\n{ocr_text}"

                        # Cleanup
                        for path in [temp_path, processed_path]:
                            if os.path.exists(path):
                                os.remove(path)

                return text

        except Exception as e:
            if self.debug:
                print(f"❌ Error processing PDF: {e}")
            return ""

    def extract_dates(self, text: str) -> Optional[datetime]:
        """
        Extracts the most probable publishing date from a block of text using a scoring system.

        This method scans the text for date-like patterns, validates them, and assigns scores
        based on context clues to determine which one is most likely to represent a publication date.

        Scoring heuristics include:
        - Presence of known publishing-related phrases (e.g., "Ausgegeben am", "Berlin, den ...")
        - Proximity to the beginning of the document (publishing dates often appear early)
        - Enclosure in parentheses (common in metadata and footnotes)

        Args:
        text (str): The input text to search for date patterns.

        Returns:
        Optional[datetime]: The most probable publishing date found in the text, or None if no valid date is found.
        """
        if self.debug:
            print("🗓️ Extracting dates...")

        # PRE-PROCESSING: Fix common OCR errors
        # Handle OCR error where "l." should be "1."
        text = re.sub(r'\bl\.\s*([a-zA-ZäöüÄÖÜß]+)', r'1. \1', text)

        if self.debug:
            print("   ✅ Applied OCR corrections")

        found_dates = []

        # Enhanced date pattern to catch more variations
        enhanced_date_patterns = [
            # Standard format: "22. September 1938"
            r'(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
            # Format with "vom": "vom 1. März 1938"
            r'vom\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
            # Format with "den": "den 27. April 1939"
            r'den\s+(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
            # Format in parentheses: "(Stand vom 22. September 1938)"
            r'\(.*?(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4}).*?\)',
        ]

        for pattern in enhanced_date_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    day_str, month_str, year_str = match.groups()
                    month_normalized = month_str.lower().strip()

                    if month_normalized in self.german_months:
                        day = int(day_str)
                        year = int(year_str)
                        month_num = self.german_months[month_normalized]
                        date_obj = datetime(year, month_num, day)

                        # Get context around the match (±150 characters)
                        context_start = max(0, match.start() - 150)
                        context_end = min(len(text), match.end() + 150)
                        context = text[context_start:context_end].lower()

                        # Initialize score
                        score = 0
                        match_text = match.group(0).lower()

                        if self.debug:
                            print(f"   Found date: {match.group(0)} -> {date_obj}")

                        # VERY HIGH PRIORITY: Document signature dates
                        very_high_priority_indicators = [
                            r'berlin,?\s+den',  # "Berlin, den" - official document signature
                            r'münchen,?\s+den',  # "München, den" - official document signature
                        ]

                        for indicator in very_high_priority_indicators:
                            if re.search(indicator, context):
                                score += 30  # Highest priority - document signature
                                if self.debug:
                                    print(f"     VERY HIGH PRIORITY: {indicator} -> +30")

                        # HIGH PRIORITY: Publishing date indicators
                        high_priority_indicators = [
                            r'stand\s+vom',  # "Stand vom"
                            r'stcmd\s+vom',  # OCR error "Stcmd vom"
                            r'\(.*?stand.*?vom',  # "(Stand vom ...)"
                            r'\(.*?stcmd.*?vom',  # "(Stcmd vom ...)"
                        ]

                        for indicator in high_priority_indicators:
                            if re.search(indicator, context):
                                score += 20  # Very high priority
                                if self.debug:
                                    print(f"     HIGH PRIORITY: {indicator} -> +20")

                        # MEDIUM PRIORITY: Official date patterns
                        medium_priority_indicators = [
                            r'ausgegeben\s+am',  # "Ausgegeben am"
                            r'verkündet\s+am',  # "Verkündet am"
                            r'wirkung\s+vom',  # "mit Wirkung vom"
                        ]

                        for indicator in medium_priority_indicators:
                            if re.search(indicator, context):
                                score += 12
                                if self.debug:
                                    print(f"     MEDIUM PRIORITY: {indicator} -> +12")

                        # LOW PRIORITY: Content-related dates (often not publishing dates)
                        low_priority_indicators = [
                            r'erlasz.*?vom',  # "Erlaß ... vom" - refers to decree dates, not publishing
                            r'erlass.*?vom',  # "Erlass ... vom" - refers to decree dates, not publishing
                        ]

                        for indicator in low_priority_indicators:
                            if re.search(indicator, context):
                                score += 5  # Lower priority - these are usually content dates
                                if self.debug:
                                    print(f"     LOW PRIORITY: {indicator} -> +5")

                        # PATTERN-SPECIFIC BONUSES
                        if 'vom' in match_text:
                            score += 8  # "vom" indicates publishing date
                            if self.debug:
                                print(f"     'vom' in match -> +8")

                        if 'den' in match_text:
                            score += 6  # "den" indicates official date
                            if self.debug:
                                print(f"     'den' in match -> +6")

                        # Parentheses bonus (official dates often in parentheses)
                        if '(' in context and ')' in context:
                            score += 10
                            if self.debug:
                                print(f"     Parentheses context -> +10")

                        # Position scoring (earlier = more likely publishing date)
                        relative_pos = match.start() / len(text)
                        if relative_pos < 0.2:  # First 20%
                            score += 8
                            if self.debug:
                                print(f"     Early position (top 20%) -> +8")
                        elif relative_pos < 0.4:  # First 40%
                            score += 4
                            if self.debug:
                                print(f"     Early position (top 40%) -> +4")

                        # Year reasonableness bonus
                        if 1920 <= year <= 1950:
                            score += 3
                        elif 1950 <= year <= 2025:
                            score += 2

                        # NEGATIVE INDICATORS (reduce score for content dates)
                        negative_indicators = [
                            r'seit\s+dem',  # "seit dem"
                            r'ab\s+dem',  # "ab dem"
                            r'erfolgte',  # "erfolgte"
                            r'geboren.*am',  # "geboren am"
                            r'verstorben.*am',  # "verstorben am"
                        ]

                        for neg_indicator in negative_indicators:
                            if re.search(neg_indicator, context):
                                score -= 5
                                if self.debug:
                                    print(f"     NEGATIVE: {neg_indicator} -> -5")

                        final_score = max(score, 0)  # Don't go negative
                        found_dates.append((date_obj, final_score, match.group(0)))

                        if self.debug:
                            print(f"     Final score: {final_score}")

                except (ValueError, KeyError) as e:
                    if self.debug:
                        print(f"     Error parsing date: {e}")
                    continue

        if not found_dates:
            if self.debug:
                print("   ❌ No valid dates found")
            return None

        # Sort by score (highest first)
        found_dates.sort(key=lambda x: x[1], reverse=True)

        if self.debug:
            print("   📊 All found dates ranked by score:")
            for i, (date_obj, score, original) in enumerate(found_dates):
                print(f"     {i + 1}. {original} -> {date_obj.strftime('%Y-%m-%d')} (score: {score})")

        # Return highest scoring date
        best_date = found_dates[0][0]
        if self.debug:
            print(f"   🏆 Selected date: {best_date.strftime('%Y-%m-%d')}")

        return best_date

    def extract_with_patterns(self, text: str, patterns: List[str], field_name: str) -> Tuple[Optional[str], float]:
        """Generic pattern-based extraction with confidence scoring"""
        candidates = []

        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle tuple matches (multiple groups)
                    if len(match) == 2:
                        result = f"{match[0]} {match[1]}"
                    else:
                        result = " ".join(match)
                else:
                    result = match

                # Clean and score
                clean_result = re.sub(r'\s+', ' ', result.strip())
                if 5 <= len(clean_result) <= 200:  # Reasonable length
                    confidence = 0.9 if len(clean_result) > 15 else 0.7
                    candidates.append((clean_result, confidence))

        if not candidates:
            return None, 0.0

        # Return best candidate
        best_match, best_score = max(candidates, key=lambda x: x[1])
        return best_match, best_score

    def extract_document_type(self, text: str) -> Tuple[Optional[str], float]:
        """Extract document type with profession enhancement"""
        best_type = None
        best_score = 0.0

        for doc_type, patterns in self.doc_type_patterns.items():
            score = 0.0
            match_count = 0

            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                if matches > 0:
                    match_count += matches
                    score += matches * 0.3

            if match_count > 0:
                # Bonus for early appearance
                early_text = text[:1000]
                if any(re.search(pattern, early_text, re.IGNORECASE) for pattern in patterns):
                    score *= 1.4

                if score > best_score:
                    best_score = score
                    best_type = doc_type

        # Try to extract profession and combine
        profession, prof_score = self.extract_with_patterns(text, self.profession_patterns, "profession")
        if profession and prof_score > 0.7 and best_type:
            best_type = f"{best_type} - {profession}"
            best_score = min(best_score + prof_score * 0.3, 1.0)

        return best_type, min(best_score, 1.0)

    def extract_all_metadata(self, text: str) -> ExtractedMetadata:
        """Extract all metadata in one optimized pass"""
        if self.debug:
            print("🔍 Extracting all metadata...")

        metadata = ExtractedMetadata()
        confidence_scores = {}

        # Extract date and year
        date_obj = self.extract_dates(text)
        if date_obj:
            metadata.date = date_obj
            metadata.year = date_obj.year
            confidence_scores['date'] = 0.9
            confidence_scores['year'] = 0.9

        # # Extract other fields
        # title, title_conf = self.extract_with_patterns(text, self.title_patterns, "title")
        # metadata.title = title
        # confidence_scores['title'] = title_conf
        #
        # publisher, pub_conf = self.extract_with_patterns(text, self.publisher_patterns, "publisher")
        # metadata.publisher = publisher
        # confidence_scores['publisher'] = pub_conf
        #
        # doc_type, type_conf = self.extract_document_type(text)
        # metadata.document_type = doc_type
        # confidence_scores['document_type'] = type_conf
        #
        # # Extract profession separately
        # profession, prof_conf = self.extract_with_patterns(text, self.profession_patterns, "profession")
        # metadata.profession = profession
        # if profession:
        #     confidence_scores['profession'] = prof_conf
        #
        # # Set confidence scores and preview
        # metadata.confidence_scores = confidence_scores
        # metadata.raw_text_preview = text[:500] + "..." if len(text) > 500 else text
        #
        # if self.debug:
        #     print("✅ Metadata extraction complete")
        #     for field, value in metadata.to_dict().items():
        #         if value and field != 'raw_text_preview':
        #             conf = confidence_scores.get(field, 0.0)
        #             print(f"   {field}: {value} (conf: {conf:.2f})")

        return metadata

    def train_ml_models(self, training_data: List[Dict]):
        """Train ML models for enhanced extraction"""
        if self.debug:
            print("🤖 Training ML models...")

        if len(training_data) < 5:
            if self.debug:
                print("⚠️ Not enough training data for ML models")
            return

        # Prepare training data
        texts = [item['text'] for item in training_data]

        # Train document type classifier
        doc_types = [item.get('document_type', '') for item in training_data]
        valid_types = [dt for dt in doc_types if dt]

        if len(valid_types) >= 3:
            self.vectorizers['doc_type'] = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
            X_tfidf = self.vectorizers['doc_type'].fit_transform(texts)

            self.label_encoders['doc_type'] = LabelEncoder()
            y_encoded = self.label_encoders['doc_type'].fit_transform(doc_types)

            self.models['doc_type'] = LogisticRegression(max_iter=1000)
            self.models['doc_type'].fit(X_tfidf, y_encoded)

            if self.debug:
                print("✅ Document type classifier trained")

    def predict_with_ml(self, text: str, field: str) -> Tuple[Optional[str], float]:
        """Predict using trained ML models"""
        if field not in self.models:
            return None, 0.0

        try:
            X_tfidf = self.vectorizers[field].transform([text])
            prediction = self.models[field].predict(X_tfidf)[0]
            probability = self.models[field].predict_proba(X_tfidf)[0].max()

            result = self.label_encoders[field].inverse_transform([prediction])[0]
            return result, probability
        except:
            return None, 0.0

    def save_models(self, filepath: str):
        """Save trained models"""
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'label_encoders': self.label_encoders
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        if self.debug:
            print(f"✅ Models saved to {filepath}")

    def load_models(self, filepath: str):
        """Load trained models"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.models = model_data['models']
            self.vectorizers = model_data['vectorizers']
            self.label_encoders = model_data['label_encoders']
            if self.debug:
                print(f"✅ Models loaded from {filepath}")
        except FileNotFoundError:
            if self.debug:
                print(f"⚠️ Model file not found: {filepath}")


class MetadataEvaluator:
    """Simplified evaluation framework"""

    @staticmethod
    def evaluate_extraction(predictions: List[ExtractedMetadata],
                          ground_truth: List[Dict]) -> Dict[str, float]:
        """Evaluate extraction performance"""
        if not predictions or not ground_truth:
            return {}

        results = {}
        fields = ['title', 'year', 'publisher', 'document_type']

        for field in fields:
            pred_values = [getattr(pred, field, None) for pred in predictions]
            true_values = [gt.get(field) for gt in ground_truth]

            # Calculate accuracy
            matches = sum(1 for p, t in zip(pred_values, true_values)
                         if p and t and str(p).lower() == str(t).lower())
            total_with_truth = sum(1 for t in true_values if t)

            accuracy = matches / total_with_truth if total_with_truth > 0 else 0.0
            results[field] = accuracy

        results['overall'] = np.mean(list(results.values()))
        return results


# # Usage example and test function
# def quick_test():
#     """Quick test function"""
#     extractor = OptimizedMetadataExtractor(debug=True)
#
#     # Test text
#     test_text = """
#     Berufs-Eignungsanforderungen
#     für den Eintritt in den Lehrberuf
#     Schmucksteinfasser
#
#     bearbeitet vom
#     Deutschen Ausschuß für Technisches Schulwesen (Datsch) E.V.
#     Berlin NW7
#
#     (Stand vom 22. September 1938)
#
#     Verlag von B.G.Teubner in Leipzig und Berlin
#     """
#
#     # Extract metadata
#     metadata = extractor.extract_all_metadata(test_text)
#
#     print("\n" + "="*60)
#     print("EXTRACTION RESULTS:")
#     print("="*60)
#
#     result_dict = metadata.to_dict()
#     for key, value in result_dict.items():
#         if value and key != 'raw_text_preview':
#             print(f"{key.upper()}: {value}")
#
#
# # Test function for date extraction with your examples
# def test_date_extraction():
#     """Test date extraction with the problematic examples"""
#     extractor = OptimizedMetadataExtractor(debug=True)
#
#     # Test case 1: berufearchiv_6322 - should extract "1. März 1938" not "19. März 1938"
#     text_6322 = """
#     BERUFSAUSBILDUNC IN DER INDUSTRlE
#     Präfungsanfordetsungen
#     fiir den Lehrberuf
#     Teppichwebsper
#
#     Am 19. März 1938 als industrieller Lelirljeruf mit-klimmt
#     durch die Reichsgruppe Industrie mul die
#     Ärbeilsgemeinscltaft tlcr Industrie· uml Hamlelsliannncrn
#     in der Reichswiktscltuktslcammet-
#
#     Slimd vmn l. Miit-Z 1938
#     """
#
#     print("=" * 60)
#     print("TEST 1: berufearchiv_6322 (should be 1. März 1938)")
#     print("=" * 60)
#     result1 = extractor.extract_dates(text_6322)
#     print(f"Result: {result1}")
#     print()
#
#     # Test case 2: berufearchiv_5526 - should find a date
#     text_5526 = """
#     Industrie-
#     Facharbeiterausbildung
#     Berufsbildungsplan
#     für den Lehrberuf
#     Schokolademacher
#     bearbeitet vom
#     Deutschen Ausschuß für Technisches Schulwesen E. V. (Datsch)
#     Berlin NW 7
#     """
#
#     print("=" * 60)
#     print("TEST 2: berufearchiv_5526 (should find a date)")
#     print("=" * 60)
#     result2 = extractor.extract_dates(text_5526)
#     print(f"Result: {result2}")
#     print()
#
#     # Test case 3: berufearchiv_5542 - should work correctly
#     text_5542 = """
#     Fachliche Vorschriften zur Regelung
#     des Lehrlingswsfms im
#     Schornsteinfegerhandwerk
#
#     Der Neichswirtschaftsniinister hat sich mit den Fachlichen
#     Vorschriften zur Regelung des Lehrlingswesens im Schornstein-
#     fegerhandwerk mit dem Erlaß IIl sW 10 861J39 vom 22. April
#     1939 einverstanden erklärt. Sie treten mit Wirkung vom 1.Juli
#     1939 in Kraft.
#
#     Mit dem Erlasz dieser Vorschriften und dem im August 1936
#     erfolgten Crlasz der Fachlichen Vorschriften für die Meister-
#     priifung verfügt das Schornftcinfegerhandwerk«nunmehr über
#     eine einheitliche Grundlage.
#
#     Berlin, den 27. April 1939.
#     """
#
#     print("=" * 60)
#     print("TEST 3: berufearchiv_5542 (should be 27. April 1939)")
#     print("=" * 60)
#     result3 = extractor.extract_dates(text_5542)
#     print(f"Result: {result3}")
#
#
# if __name__ == "__main__":
#     # Run the original test
#     quick_test()
#
#     print("\n" + "="*80)
#     print("TESTING DATE EXTRACTION WITH PROBLEM CASES")
#     print("="*80)
#
#     # Run date extraction tests
#     test_date_extraction()