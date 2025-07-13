"""
Optimized Metadata Extraction System for Historical German Documents
Consolidated and optimized version with improved performance
"""

import os
import pickle
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import IPython
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder


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
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}schmidt)',  # Goldschmidt, etc.
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}bauer)',  # Instrumentenbauer, etc.
            r'(Kaufmann|Mechaniker|Elektriker|Bäcker|Schneider|Tischler)'
        ]

        # Date publishing indicators
        self.publishing_indicators = [
            r'(?:Stand\s+vom|Stcmd\s+vom)',
            r'(?:Berlin,?\s+den)',
            r'(?:München,?\s+den)',
            r'(?:Datum\s*:)',
            r'(?:Ausgegeben\s+am)',
            r'(?:Verkündet\s+am)'
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
                    pages = convert_from_path(pdf_path, first_page=i + 1, last_page=i + 1, dpi=300)
                    if pages:
                        # Save temporary image
                        temp_path = f"temp_page_{i + 1}.png"
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

        found_dates = []  # Will store tuples of (datetime object, score)

        # Iterate over all regex matches of the date pattern in the text
        for match in self.date_pattern.finditer(text):
            # IPython.embed()  # Useful for debugging interactively; you can remove/comment this in production

            # Extract day, month, year from the regex groups
            day_str, month_str, year_str = match.groups()
            month_normalized = month_str.lower().strip()

            if month_normalized in self.german_months:
                try:
                    # Convert string values to integers
                    day = int(day_str)
                    year = int(year_str)
                    month_num = self.german_months[month_normalized]  # Convert German month to month number

                    # Construct a datetime object from the extracted values
                    date_obj = datetime(year, month_num, day)

                    # ---- Begin Scoring Heuristics ---- #
                    score = 0

                    # 1. Extract context: 100 characters before and after the match
                    context = text[max(0, match.start() - 100):match.end() + 100].lower()

                    # 2. Check for publishing-related phrases (e.g., "Berlin, den", "Verkündet am")
                    for indicator in self.publishing_indicators:
                        if re.search(indicator, context, re.IGNORECASE):
                            score += 10

                    # 3. Position in document: earlier dates are more likely to be the publishing date
                    relative_pos = match.start() / len(text)
                    if relative_pos < 0.3:
                        score += 5

                    # 4. Parentheses often wrap dates in footnotes, headers, etc.
                    if '(' in context and ')' in context:
                        score += 6

                    # Save the date and its associated score
                    found_dates.append((date_obj, score))

                except (ValueError, KeyError):
                    # Skip this match if parsing fails (e.g., invalid date or unknown month)
                    continue

        # No dates found
        if not found_dates:
            return None

        # Return the date with the highest score
        return max(found_dates, key=lambda x: x[1])[0]

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


