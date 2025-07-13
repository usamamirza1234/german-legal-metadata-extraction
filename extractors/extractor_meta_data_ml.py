"""
Machine Learning Training Pipeline for German Legal Document Metadata Extraction
Thesis Implementation - Advanced ML Models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle
import json
from typing import List, Dict, Tuple
import re


class MLMetadataExtractor:
    """
    Machine Learning pipeline for metadata extraction
    Combines rule-based and ML approaches
    """

    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}
        self.feature_extractors = {}

    def create_features(self, text: str) -> Dict[str, float]:
        """Extract features for ML models"""
        features = {}

        # Text statistics
        features['text_length'] = len(text)
        features['line_count'] = len(text.split('\n'))
        features['word_count'] = len(text.split())

        # Position-based features
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Document structure features
        features['has_paragraphs'] = 1 if '§' in text else 0
        features['has_date_pattern'] = 1 if re.search(r'\d{1,2}\.\s*\w+\s*\d{4}', text) else 0
        features['has_city_date'] = 1 if re.search(r'Berlin.*den|München.*den', text) else 0

        # Keyword density features
        keywords = {
            'prufung': ['prüfung', 'prüfungsordnung', 'prüfungsanforderungen'],
            'ausbildung': ['ausbildung', 'ausbildungsordnung', 'lehrplan'],
            'ministerium': ['ministerium', 'reichsministerium', 'bundesministerium'],
            'verordnung': ['verordnung', 'anordnung', 'bestimmungen'],
            'handel': ['handel', 'kaufmann', 'verkäufer', 'handels'],
            'handwerk': ['handwerk', 'meister', 'geselle', 'handwerks']
        }

        text_lower = text.lower()
        for category, words in keywords.items():
            count = sum(text_lower.count(word) for word in words)
            features[f'{category}_density'] = count / len(text.split()) if text.split() else 0

        # Title candidate features (for title extraction)
        if lines:
            first_line = lines[0]
            features['first_line_length'] = len(first_line)
            features['first_line_capitalized'] = 1 if first_line[0].isupper() else 0
            features['first_line_has_keywords'] = 1 if any(kw in first_line.lower()
                                                          for kw_list in keywords.values()
                                                          for kw in kw_list) else 0

        return features

    def prepare_training_data(self, labeled_documents: List[Dict]) -> Tuple[pd.DataFrame, Dict]:
        """Prepare training data for ML models"""

        training_data = []

        for doc in labeled_documents:
            text = doc['text']
            metadata = doc['metadata']

            # Extract features
            features = self.create_features(text)

            # Add labels
            features.update({
                'title': metadata.get('title', ''),
                'year': metadata.get('year', None),
                'publisher': metadata.get('publisher', ''),
                'document_type': metadata.get('document_type', ''),
                'author': metadata.get('author', '')
            })

            # Add text for TF-IDF
            features['text'] = text

            training_data.append(features)

        df = pd.DataFrame(training_data)

        # Separate features and targets
        feature_columns = [col for col in df.columns if col not in ['title', 'year', 'publisher', 'document_type', 'author', 'text']]

        targets = {
            'title': df['title'].fillna(''),
            'year': df['year'].fillna(0),
            'publisher': df['publisher'].fillna(''),
            'document_type': df['document_type'].fillna(''),
            'author': df['author'].fillna('')
        }

        return df[feature_columns + ['text']], targets

    def train_title_extractor(self, df: pd.DataFrame, titles: pd.Series):
        """Train model to identify if a line is likely a title"""

        # Create line-level training data
        line_features = []
        line_labels = []

        for idx, (text, title) in enumerate(zip(df['text'], titles)):
            lines = [line.strip() for line in text.split('\n') if line.strip()]

            for line_idx, line in enumerate(lines[:20]):  # First 20 lines only
                # Features for this line
                line_feat = {
                    'line_position': line_idx,
                    'line_length': len(line),
                    'is_capitalized': 1 if line[0].isupper() else 0,
                    'word_count': len(line.split()),
                    'has_colon': 1 if ':' in line else 0,
                    'has_paragraph': 1 if '§' in line else 0,
                    'relative_position': line_idx / len(lines) if lines else 0
                }

                # Keyword features
                line_lower = line.lower()
                line_feat['has_prufung'] = 1 if 'prüfung' in line_lower else 0
                line_feat['has_verordnung'] = 1 if 'verordnung' in line_lower else 0
                line_feat['has_ordnung'] = 1 if 'ordnung' in line_lower else 0

                # Label: 1 if this line matches the title, 0 otherwise
                is_title = 1 if title and title.strip() in line else 0

                line_features.append(line_feat)
                line_labels.append(is_title)

        # Train Random Forest classifier
        feature_df = pd.DataFrame(line_features)
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df, line_labels, test_size=0.2, random_state=42
        )

        self.models['title_classifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['title_classifier'].fit(X_train, y_train)

        # Evaluate
        y_pred = self.models['title_classifier'].predict(X_test)
        print(f"Title Classifier Accuracy: {accuracy_score(y_test, y_pred):.3f}")

        return self.models['title_classifier']

    def train_document_type_classifier(self, df: pd.DataFrame, doc_types: pd.Series):
        """Train document type classifier"""

        # Filter out empty document types
        valid_mask = doc_types != ''
        valid_texts = df[valid_mask]['text']
        valid_types = doc_types[valid_mask]

        if len(valid_types) < 10:
            print("⚠️ Not enough labeled document types for training")
            return None

        # TF-IDF vectorization
        self.vectorizers['doc_type'] = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words=None  # Keep German stop words for now
        )

        X_tfidf = self.vectorizers['doc_type'].fit_transform(valid_texts)

        # Label encoding
        self.label_encoders['doc_type'] = LabelEncoder()
        y_encoded = self.label_encoders['doc_type'].fit_transform(valid_types)

        # Train classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X_tfidf, y_encoded, test_size=0.2, random_state=42
        )

        self.models['doc_type_classifier'] = LogisticRegression(max_iter=1000)
        self.models['doc_type_classifier'].fit(X_train, y_train)

        # Evaluate
        y_pred = self.models['doc_type_classifier'].predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Document Type Classifier Accuracy: {accuracy:.3f}")

        # Print classification report
        target_names = self.label_encoders['doc_type'].classes_
        print("\nDocument Type Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        return self.models['doc_type_classifier']

    def train_publisher_extractor(self, df: pd.DataFrame, publishers: pd.Series):
        """Train publisher/institution extractor"""

        # Similar approach to document type but with NER-style features
        valid_mask = publishers != ''
        valid_texts = df[valid_mask]['text']
        valid_publishers = publishers[valid_mask]

        if len(valid_publishers) < 5:
            print("⚠️ Not enough labeled publishers for training")
            return None

        # Extract institution patterns
        institution_features = []
        institution_labels = []

        # Common German institution patterns
        institution_patterns = [
            r'((?:Reichs|Bundes)?ministerium[^.\n]{0,50})',
            r'((?:Reichs|Bundes)?amt[^.\n]{0,30})',
            r'(Industrie.*?Handelskammer[^.\n]{0,30})',
            r'([A-ZÄÖÜ][^.\n]*(?:ministerium|amt|kammer)[^.\n]*)'
        ]

        for text, true_publisher in zip(valid_texts, valid_publishers):
            # Find all potential institutions
            candidates = []
            for pattern in institution_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                candidates.extend(matches)

            # Create features for each candidate
            for candidate in candidates:
                feat = {
                    'candidate_length': len(candidate),
                    'position_in_text': text.find(candidate) / len(text) if text else 0,
                    'has_reichs': 1 if 'reichs' in candidate.lower() else 0,
                    'has_bundes': 1 if 'bundes' in candidate.lower() else 0,
                    'has_ministerium': 1 if 'ministerium' in candidate.lower() else 0,
                    'has_amt': 1 if 'amt' in candidate.lower() else 0,
                    'has_kammer': 1 if 'kammer' in candidate.lower() else 0
                }

                # Label: 1 if this candidate matches the true publisher
                is_correct = 1 if true_publisher.lower() in candidate.lower() or candidate.lower() in true_publisher.lower() else 0

                institution_features.append(feat)
                institution_labels.append(is_correct)

        if not institution_features:
            print("⚠️ No institution candidates found for training")
            return None

        # Train classifier
        feature_df = pd.DataFrame(institution_features)
        X_train, X_test, y_train, y_test = train_test_split(
            feature_df, institution_labels, test_size=0.2, random_state=42
        )

        self.models['publisher_classifier'] = RandomForestClassifier(n_estimators=100, random_state=42)
        self.models['publisher_classifier'].fit(X_train, y_train)

        # Evaluate
        y_pred = self.models['publisher_classifier'].predict(X_test)
        print(f"Publisher Classifier Accuracy: {accuracy_score(y_test, y_pred):.3f}")

        return self.models['publisher_classifier']

    def train_all_models(self, labeled_documents: List[Dict]):
        """Train all ML models"""
        print("🤖 TRAINING ALL ML MODELS...")

        # Prepare data
        df, targets = self.prepare_training_data(labeled_documents)

        print(f"📊 Training data: {len(df)} documents")

        # Train individual models
        print("\n1. Training Title Extractor...")
        self.train_title_extractor(df, targets['title'])

        print("\n2. Training Document Type Classifier...")
        self.train_document_type_classifier(df, targets['document_type'])

        print("\n3. Training Publisher Extractor...")
        self.train_publisher_extractor(df, targets['publisher'])

        print("\n✅ All models trained successfully!")

    def predict_title(self, text: str) -> Tuple[str, float]:
        """Predict title using trained model"""
        if 'title_classifier' not in self.models:
            return None, 0.0

        lines = [line.strip() for line in text.split('\n') if line.strip()]
        best_line = ""
        best_score = 0.0

        for line_idx, line in enumerate(lines[:20]):
            # Create features for this line
            line_feat = {
                'line_position': line_idx,
                'line_length': len(line),
                'is_capitalized': 1 if line[0].isupper() else 0,
                'word_count': len(line.split()),
                'has_colon': 1 if ':' in line else 0,
                'has_paragraph': 1 if '§' in line else 0,
                'relative_position': line_idx / len(lines) if lines else 0,
                'has_prufung': 1 if 'prüfung' in line.lower() else 0,
                'has_verordnung': 1 if 'verordnung' in line.lower() else 0,
                'has_ordnung': 1 if 'ordnung' in line.lower() else 0
            }

            # Predict probability
            feature_df = pd.DataFrame([line_feat])
            prob = self.models['title_classifier'].predict_proba(feature_df)[0][1]  # Probability of being title

            if prob > best_score:
                best_score = prob
                best_line = line

        return best_line if best_score > 0.5 else None, best_score

    def predict_document_type(self, text: str) -> Tuple[str, float]:
        """Predict document type using trained model"""
        if 'doc_type_classifier' not in self.models:
            return None, 0.0

        # Vectorize text
        X_tfidf = self.vectorizers['doc_type'].transform([text])

        # Predict
        prediction = self.models['doc_type_classifier'].predict(X_tfidf)[0]
        probability = self.models['doc_type_classifier'].predict_proba(X_tfidf)[0].max()

        # Decode label
        doc_type = self.label_encoders['doc_type'].inverse_transform([prediction])[0]

        return doc_type, probability

    def predict_publisher(self, text: str) -> Tuple[str, float]:
        """Predict publisher using trained model"""
        if 'publisher_classifier' not in self.models:
            return None, 0.0

        # Extract institution candidates
        institution_patterns = [
            r'((?:Reichs|Bundes)?ministerium[^.\n]{0,50})',
            r'((?:Reichs|Bundes)?amt[^.\n]{0,30})',
            r'(Industrie.*?Handelskammer[^.\n]{0,30})',
            r'([A-ZÄÖÜ][^.\n]*(?:ministerium|amt|kammer)[^.\n]*)'
        ]

        candidates = []
        for pattern in institution_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            candidates.extend(matches)

        if not candidates:
            return None, 0.0

        # Score each candidate
        best_candidate = ""
        best_score = 0.0

        for candidate in candidates:
            feat = {
                'candidate_length': len(candidate),
                'position_in_text': text.find(candidate) / len(text) if text else 0,
                'has_reichs': 1 if 'reichs' in candidate.lower() else 0,
                'has_bundes': 1 if 'bundes' in candidate.lower() else 0,
                'has_ministerium': 1 if 'ministerium' in candidate.lower() else 0,
                'has_amt': 1 if 'amt' in candidate.lower() else 0,
                'has_kammer': 1 if 'kammer' in candidate.lower() else 0
            }

            # Predict probability
            feature_df = pd.DataFrame([feat])
            prob = self.models['publisher_classifier'].predict_proba(feature_df)[0][1]

            if prob > best_score:
                best_score = prob
                best_candidate = candidate.strip()

        return best_candidate if best_score > 0.5 else None, best_score

    def save_models(self, filepath: str):
        """Save all trained models and preprocessing objects"""
        model_data = {
            'models': self.models,
            'vectorizers': self.vectorizers,
            'label_encoders': self.label_encoders
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"✅ Models saved to {filepath}")

    def load_models(self, filepath: str):
        """Load trained models and preprocessing objects"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.models = model_data['models']
        self.vectorizers = model_data['vectorizers']
        self.label_encoders = model_data['label_encoders']

        print(f"✅ Models loaded from {filepath}")


class DatasetCreator:
    """Helper class to create training datasets"""

    @staticmethod
    def create_sample_dataset() -> List[Dict]:
        """Create a sample dataset for testing"""
        return [
            {
                'text': """
                Verordnung über die Prüfungsanforderungen für Kaufleute
                
                Reichsministerium für Wirtschaft
                Berlin, den 27. April 1939
                
                Auf Grund des § 45 der Handwerksordnung wird hiermit bestimmt:
                
                § 1
                Die Prüfung für Kaufleute umfasst folgende Gebiete:
                1. Buchführung und Bilanzwesen
                2. Handelskunde
                3. Rechtskunde
                """,
                'metadata': {
                    'title': 'Verordnung über die Prüfungsanforderungen für Kaufleute',
                    'year': 1939,
                    'publisher': 'Reichsministerium für Wirtschaft',
                    'document_type': 'Prüfungsordnung',
                    'author': None
                }
            },
            {
                'text': """
                Lehrplan für den Unterricht in der Berufsschule
                Fachrichtung: Maschinenbau
                
                Preußisches Ministerium für Wissenschaft, Kunst und Volksbildung
                
                § 1 Allgemeine Bestimmungen
                Der Unterricht in der Berufsschule für Maschinenbau gliedert sich in:
                1. Fachkunde
                2. Fachzeichnen
                3. Fachrechnen
                
                Stand vom 15. März 1925
                """,
                'metadata': {
                    'title': 'Lehrplan für den Unterricht in der Berufsschule Fachrichtung: Maschinenbau',
                    'year': 1925,
                    'publisher': 'Preußisches Ministerium für Wissenschaft, Kunst und Volksbildung',
                    'document_type': 'Lehrplan',
                    'author': None
                }
            },
            {
                'text': """
                Ausbildungsordnung für den Beruf des Elektrikers
                
                Bundesministerium für Bildung und Forschung
                Bonn, den 12. Juni 1987
                
                Aufgrund des § 25 des Berufsbildungsgesetzes wird verordnet:
                
                § 1 Staatliche Anerkennung des Ausbildungsberufes
                Der Ausbildungsberuf Elektriker wird staatlich anerkannt.
                
                § 2 Ausbildungsdauer
                Die Ausbildung dauert 3,5 Jahre.
                """,
                'metadata': {
                    'title': 'Ausbildungsordnung für den Beruf des Elektrikers',
                    'year': 1987,
                    'publisher': 'Bundesministerium für Bildung und Forschung',
                    'document_type': 'Ausbildungsordnung',
                    'author': None
                }
            }
        ]

    @staticmethod
    def load_from_json(filepath: str) -> List[Dict]:
        """Load training dataset from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def save_to_json(dataset: List[Dict], filepath: str):
        """Save training dataset to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)


# Example usage and evaluation
if __name__ == "__main__":
    # Initialize ML extractor
    ml_extractor = MLMetadataExtractor()

    # Create or load training dataset
    dataset_creator = DatasetCreator()
    training_data = dataset_creator.create_sample_dataset()

    print(f"📊 Training dataset size: {len(training_data)} documents")

    # Train models
    ml_extractor.train_all_models(training_data)

    # Test on new document
    test_text = """
    Bestimmungen für die Gesellenprüfung im Bäckerhandwerk
    
    Handwerkskammer München
    München, den 3. November 1952
    
    § 1 Prüfungsgegenstand
    Die Gesellenprüfung erstreckt sich auf:
    1. Praktische Arbeiten
    2. Fachkunde
    """

    print("\n" + "="*60)
    print("TESTING TRAINED MODELS")
    print("="*60)

    # Test predictions
    title, title_conf = ml_extractor.predict_title(test_text)
    doc_type, type_conf = ml_extractor.predict_document_type(test_text)
    publisher, pub_conf = ml_extractor.predict_publisher(test_text)

    print(f"Predicted Title: {title} (confidence: {title_conf:.3f})")
    print(f"Predicted Document Type: {doc_type} (confidence: {type_conf:.3f})")
    print(f"Predicted Publisher: {publisher} (confidence: {pub_conf:.3f})")

    # Save models
    ml_extractor.save_models('trained_models.pkl')

    print("\n✅ Training and testing complete!")