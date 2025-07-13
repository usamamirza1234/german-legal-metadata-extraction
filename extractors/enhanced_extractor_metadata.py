"""
Enhanced Metadata Extraction for Historical German Legal Documents
Thesis Implementation - Vocational Education Regulations (1920-2025)
"""

import re
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

from extractors.extractor_metadata import ExtractedMetadata


class EnhancedMetadataExtractor:
    """
    Enhanced metadata extractor specifically for German vocational education documents
    Combines rule-based and ML approaches
    """

    def __init__(self, debug=False):
        self.debug = debug
        self.trained_models = {}

        # Initialize your existing extractors
        from extractors.extractor_german_date import ExtractorGermanDate
        self.german_date_extractor = ExtractorGermanDate(debug)

        # Load spaCy model for German
        try:
            self.nlp = spacy.load("de_core_news_sm")
        except OSError:
            print("⚠️ German spaCy model not found. Install with: python -m spacy download de_core_news_sm")
            self.nlp = None

        # Initialize pattern libraries
        self._init_patterns()

    def _init_patterns(self):
        """Initialize regex patterns for German legal documents"""

        # Title patterns - German legal documents often have specific structures
        self.title_patterns = [
            r'(?:Verordnung|Anordnung|Gesetz|Bestimmungen|Richtlinien)\s+(?:über|für|zur|betreffend)\s+([^.\n]{20,100})',
            r'([A-ZÄÖÜ][^.\n]{20,100}(?:verordnung|anordnung|gesetz|bestimmungen|richtlinien))',
            r'^([A-ZÄÖÜ][^.\n]{15,80})$'  # Capitalized lines (potential titles)
        ]

        # Publisher/Institution patterns for vocational education
        self.publisher_patterns = [
            r'(Reichsministerium\s+für\s+[^.\n]+)',
            r'(Bundesministerium\s+für\s+[^.\n]+)',
            r'(Ministerium\s+für\s+[^.\n]+)',
            r'(Preußisches?\s+Ministerium\s+[^.\n]+)',
            r'(Kultusministerium\s+[^.\n]*)',
            r'(Handelsministerium\s+[^.\n]*)',
            r'(Reichsamt\s+[^.\n]+)',
            r'(Bundesamt\s+[^.\n]+)',
            r'(Deutscher\s+Industrie-\s*und\s*Handelskammertag)',
            r'(Industrie-\s*und\s*Handelskammer\s+[^.\n]*)',
        ]

        # Document type patterns specific to vocational education
        self.doc_type_patterns = {
            'Prüfungsordnung': [
                r'\bPrüfungsordnung\b',
                r'\bPrüfungsanforderungen\b',
                r'Ordnung.*Prüfung',
                r'Bestimmungen.*Prüfung'
            ],
            'Lehrplan': [
                r'\bLehrplan\b',
                r'\bLehrpläne\b',
                r'Plan.*Unterricht',
                r'Unterrichtsplan'
            ],
            'Ausbildungsordnung': [
                r'\bAusbildungsordnung\b',
                r'Ordnung.*Ausbildung',
                r'Bestimmungen.*Ausbildung'
            ],
            'Verordnung': [
                r'\bVerordnung\b(?!\s*über\s*Prüfung)',  # General regulation
                r'\bAnordnung\b'
            ],
            'Gesetz': [
                r'\bGesetz\b',
                r'\bBerufsbildungsgesetz\b',
                r'\bHandwerksordnung\b'
            ],
            'Richtlinien': [
                r'\bRichtlinien\b',
                r'\bBestimmungen\b'
            ]
        }

        # Author patterns (often missing in legal docs, but sometimes present)
        self.author_patterns = [
            r'(?:verfasst|erstellt|bearbeitet)\s+von\s+([^.\n]+)',
            r'Autor:?\s*([^.\n]+)',
            r'Bearbeiter:?\s*([^.\n]+)',
            r'(?:Dr\.|Prof\.)\s*([A-ZÄÖÜa-zäöüß\s]+)'
        ]

        # Vocational field patterns
        self.vocational_fields = [
            'Handel', 'Handwerk', 'Industrie', 'Landwirtschaft', 'Technik',
            'Kaufmann', 'Mechaniker', 'Schlosser', 'Elektriker', 'Bäcker',
            'Fleischer', 'Schneider', 'Tischler', 'Maurer', 'Gärtner'
        ]

    def extract_title(self, text: str) -> Tuple[Optional[str], float]:
        """Extract document title with confidence score"""
        if self.debug:
            print("📝 Extracting TITLE...")

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        candidates = []

        # Pattern-based extraction
        for pattern in self.title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                candidates.append((match.strip(), 0.8))

        # Heuristic-based extraction from document structure
        for i, line in enumerate(lines[:10]):  # Check first 10 lines
            score = 0.0

            # Length heuristic
            if 20 <= len(line) <= 100:
                score += 0.3
            elif 15 <= len(line) <= 150:
                score += 0.2

            # Position heuristic (earlier = more likely title)
            score += max(0, (10 - i) * 0.05)

            # Capitalization heuristic
            if line[0].isupper():
                score += 0.2

            # Content heuristic - vocational education keywords
            voc_keywords = ['Prüfung', 'Ausbildung', 'Lehrplan', 'Ordnung', 'Bestimmungen']
            if any(keyword in line for keyword in voc_keywords):
                score += 0.3

            # Avoid obvious non-titles
            if any(avoid in line.lower() for avoid in ['seite', 'page', 'stand vom']):
                score -= 0.5

            if score > 0.4:
                candidates.append((line, score))

        if not candidates:
            return None, 0.0

        # Return highest scoring candidate
        best_title, best_score = max(candidates, key=lambda x: x[1])

        if self.debug:
            print(f"   🏆 Best title: '{best_title}' (score: {best_score:.2f})")

        return best_title, best_score

    def extract_publisher(self, text: str) -> Tuple[Optional[str], float]:
        """Extract publisher/institution with confidence score"""
        if self.debug:
            print("🏢 Extracting PUBLISHER...")

        candidates = []

        for pattern in self.publisher_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                clean_match = re.sub(r'\s+', ' ', match.strip())
                confidence = 0.9 if len(clean_match) > 10 else 0.7
                candidates.append((clean_match, confidence))

        if not candidates:
            # Fallback: look for any institution-like entities
            institution_keywords = ['ministerium', 'amt', 'kammer', 'reichs', 'bundes']
            lines = text.split('\n')

            for line in lines[:20]:  # Check first 20 lines
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in institution_keywords):
                    if 10 <= len(line.strip()) <= 80:
                        candidates.append((line.strip(), 0.5))

        if not candidates:
            return None, 0.0

        # Return highest scoring candidate
        best_publisher, best_score = max(candidates, key=lambda x: x[1])

        if self.debug:
            print(f"   🏆 Best publisher: '{best_publisher}' (score: {best_score:.2f})")

        return best_publisher, best_score

    def extract_document_type(self, text: str) -> Tuple[Optional[str], float]:
        """Extract document type with confidence score"""
        if self.debug:
            print("📄 Extracting DOCUMENT TYPE...")

        candidates = []

        for doc_type, patterns in self.doc_type_patterns.items():
            type_score = 0.0
            match_count = 0

            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                if matches > 0:
                    match_count += matches
                    type_score += matches * 0.2

            if match_count > 0:
                # Bonus for multiple mentions
                if match_count > 2:
                    type_score *= 1.3

                # Bonus for early appearance in document
                early_text = text[:1000]
                if any(re.search(pattern, early_text, re.IGNORECASE) for pattern in patterns):
                    type_score *= 1.2

                candidates.append((doc_type, min(type_score, 1.0)))

        if not candidates:
            return None, 0.0

        # Return highest scoring candidate
        best_type, best_score = max(candidates, key=lambda x: x[1])

        if self.debug:
            print(f"   🏆 Best document type: '{best_type}' (score: {best_score:.2f})")

        return best_type, best_score

    def extract_author(self, text: str) -> Tuple[Optional[str], float]:
        """Extract author with confidence score (often not present in legal docs)"""
        if self.debug:
            print("👤 Extracting AUTHOR...")

        candidates = []

        for pattern in self.author_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]
                clean_match = re.sub(r'\s+', ' ', match.strip())
                if 3 <= len(clean_match) <= 50:  # Reasonable name length
                    candidates.append((clean_match, 0.7))

        if not candidates:
            return None, 0.0

        # Return highest scoring candidate
        best_author, best_score = max(candidates, key=lambda x: x[1])

        if self.debug:
            print(f"   🏆 Best author: '{best_author}' (score: {best_score:.2f})")

        return best_author, best_score

    def extract_vocational_field(self, text: str) -> Tuple[Optional[str], float]:
        """Extract vocational field/trade area"""
        if self.debug:
            print("⚙️ Extracting VOCATIONAL FIELD...")

        field_scores = {}
        text_lower = text.lower()

        for field in self.vocational_fields:
            field_lower = field.lower()
            # Count occurrences
            count = len(re.findall(rf'\b{re.escape(field_lower)}\w*', text_lower))
            if count > 0:
                field_scores[field] = count * 0.2

        if not field_scores:
            return None, 0.0

        best_field = max(field_scores.items(), key=lambda x: x[1])
        return best_field[0], min(best_field[1], 1.0)

    def extract_all_metadata(self, text: str) -> ExtractedMetadata:
        """Extract all metadata from text"""
        if self.debug:
            print("🔍 EXTRACTING ALL METADATA...")
            print(f"   📄 Text length: {len(text)} characters")

        metadata = ExtractedMetadata()
        confidence_scores = {}

        # Extract date using existing extractor
        date_result = self.german_date_extractor.find_publishing_date_with_details(text)
        if date_result:
            metadata.date = date_result['date']
            metadata.year = date_result['date'].year
            confidence_scores['date'] = 0.9 if date_result['confidence'] == 'high' else 0.6

        # Extract other metadata
        title, title_conf = self.extract_title(text)
        metadata.title = title
        confidence_scores['title'] = title_conf

        publisher, pub_conf = self.extract_publisher(text)
        metadata.publisher = publisher
        confidence_scores['publisher'] = pub_conf

        doc_type, type_conf = self.extract_document_type(text)
        metadata.document_type = doc_type
        confidence_scores['document_type'] = type_conf

        author, auth_conf = self.extract_author(text)
        metadata.author = author
        confidence_scores['author'] = auth_conf

        # Additional field for vocational education documents
        voc_field, voc_conf = self.extract_vocational_field(text)
        if voc_field:
            if not metadata.document_type:
                metadata.document_type = f"{voc_field} Document"
            confidence_scores['vocational_field'] = voc_conf

        metadata.confidence_scores = confidence_scores
        metadata.raw_text_preview = text[:500] + "..." if len(text) > 500 else text

        if self.debug:
            print("✅ METADATA EXTRACTION COMPLETE")
            print(f"   Title: {metadata.title}")
            print(f"   Year: {metadata.year}")
            print(f"   Publisher: {metadata.publisher}")
            print(f"   Document Type: {metadata.document_type}")
            print(f"   Author: {metadata.author}")
            print(f"   Confidence Scores: {confidence_scores}")

        return metadata

    def train_ml_models(self, training_data: List[Dict]):
        """Train ML models for metadata extraction (for future enhancement)"""
        if self.debug:
            print("🤖 TRAINING ML MODELS...")

        # Prepare training data
        texts = [item['text'] for item in training_data]

        # Train title classifier
        titles = [item.get('title', '') for item in training_data]
        if any(titles):
            # TF-IDF vectorization + Naive Bayes (simple example)
            vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
            X = vectorizer.fit_transform(texts)

            # This is a simplified example - you'd need proper labels for classification
            self.trained_models['title_vectorizer'] = vectorizer

        if self.debug:
            print("✅ ML model training complete")

    def save_models(self, filepath: str):
        """Save trained models"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.trained_models, f)

    def load_models(self, filepath: str):
        """Load trained models"""
        with open(filepath, 'rb') as f:
            self.trained_models = pickle.load(f)


# # Example usage and testing
# if __name__ == "__main__":
#     # Initialize extractor
#     extractor = EnhancedMetadataExtractor(debug=True)
#
#     # Example German legal document text
#     sample_text = """
#     Verordnung über die Prüfungsanforderungen für Kaufleute
#
#     Reichsministerium für Wirtschaft
#     Berlin, den 27. April 1939
#
#     Auf Grund des § 45 der Handwerksordnung wird hiermit bestimmt:
#
#     § 1
#     Die Prüfung für Kaufleute umfasst folgende Gebiete:
#     1. Buchführung und Bilanzwesen
#     2. Handelskunde
#     3. Rechtskunde
#
#     (Stand vom 1. Mai 1939)
#     """
#
#     # Extract metadata
#     metadata = extractor.extract_all_metadata(sample_text)
#
#     # Display results
#     print("\n" + "=" * 60)
#     print("EXTRACTED METADATA RESULTS:")
#     print("=" * 60)
#
#     result_dict = metadata.to_dict()
#     for key, value in result_dict.items():
#         if value is not None and key != 'raw_text_preview':
#             print(f"{key.upper()}: {value}")
#
#     print("\nCONFIDENCE SCORES:")
#     for field, score in metadata.confidence_scores.items():
#         print(f"  {field}: {score:.2f}")