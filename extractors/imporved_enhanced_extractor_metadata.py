"""
Improved Metadata Extractors Based on Real OCR Test Results
Addresses specific issues found in the Schmucksteinfasser document
"""

import re
from typing import List, Dict, Tuple, Optional
from enhanced_extractor_metadata import EnhancedMetadataExtractor


class ImprovedMetadataExtractor(EnhancedMetadataExtractor):
    """
    Enhanced version that addresses specific issues found in testing
    """

    def __init__(self, debug=False):
        super().__init__(debug)

        # Enhanced patterns based on test results
        self._init_enhanced_patterns()

    def _init_enhanced_patterns(self):
        """Initialize enhanced patterns based on real document analysis"""

        # Enhanced title patterns for compound titles
        self.enhanced_title_patterns = [
            # Multi-line titles (common in historical documents)
            r'(?:Berufs-?\s*)?([A-ZÄÖÜ][a-zäöüß]*anforderungen)\s*(?:\n.*?)?\s*(?:für|fü r)\s+([^.\n]{10,60})',

            # Title with profession specification
            r'(Berufs-?[Ee]ignungsanforderungen)\s*(?:\n.*?)?\s*(?:für.*?)\s*([A-ZÄÖÜ][a-zäöüß]{8,25})',

            # Simple title patterns with OCR error tolerance
            r'^([A-ZÄÖÜ][a-zäöüß\-\s]{8,50}anforderungen)',

            # Title spanning multiple lines
            r'([A-ZÄÖÜ][a-zäöüß\-]*)\s*\n\s*([A-ZÄÖÜ][a-zäöüß]*anforderungen)',
        ]

        # Enhanced publisher patterns for German institutions
        self.enhanced_publisher_patterns = [
            # German committee/association patterns
            r'(Deutscher\s+Ausschuss?\s+für\s+[^.\n]{5,40})\s*(?:\([^)]+\))?\s*[Ee]\.?[Vv]\.?',
            r'(Deutscher\s+Ausschuß\s+für\s+[^.\n]{5,40})\s*(?:\([^)]+\))?\s*[Ee]\.?[Vv]\.?',

            # Institution with location
            r'([A-ZÄÖÜ][^.\n]*(?:Ausschuss|Ausschuß|Kammer|Ministerium)[^.\n]*)\s*[·•]\s*Berlin',

            # Working group patterns
            r'(Arbeitsgemeinschaft\s+[^.\n]{10,50})',

            # Reich organization patterns
            r'(Reichsgruppe\s+[A-ZÄÖÜ][a-zäöüß]+)',
            r'(Reichs[a-zäöüß]+kammer)',

            # Publishers with location
            r'Verlag\s+von\s+([^.\n]{5,30})\s+in\s+[A-ZÄÖÜ][a-zäöüß]+',
        ]

        # Enhanced document type patterns for vocational documents
        self.enhanced_doc_type_patterns = {
            'Eignungsanforderungen': [
                r'[Ee]ignungsanforderungen',
                r'Berufs-?[Ee]ignungsanforderungen',
                r'Anforderungen.*Eintritt.*Lehrberuf',
            ],
            'Berufsanforderungen': [
                r'Berufsanforderungen',
                r'Anforderungen.*Beruf',
            ],
            'Lehrplan': [
                r'Lehrplan',
                r'Unterrichtsplan',
            ],
            'Prüfungsordnung': [
                r'Prüfungsordnung',
                r'Prüfungsanforderungen',
            ],
            'Ausbildungsordnung': [
                r'Ausbildungsordnung',
                r'Ordnung.*Ausbildung',
            ]
        }

        # Enhanced profession/trade patterns
        self.enhanced_profession_patterns = [
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}fasser)',  # Steinfasser, etc.
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}macher)',  # Uhrmacher, etc.
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}schmidt)',  # Goldschmidt, etc.
            r'([A-ZÄÖÜ][a-zäöüß]{6,20}bauer)',   # Instrumentenbauer, etc.
            r'(Kaufmann)',
            r'(Mechaniker)',
            r'(Elektriker)',
        ]

    def extract_enhanced_title(self, text: str) -> Tuple[Optional[str], float]:
        """Enhanced title extraction handling multi-line and compound titles"""
        if self.debug:
            print("📝 ENHANCED TITLE EXTRACTION...")

        candidates = []

        # Try enhanced patterns first
        for pattern in self.enhanced_title_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    # Combine tuple elements
                    if len(match) == 2:
                        title_candidate = f"{match[0]} {match[1]}"
                    else:
                        title_candidate = " ".join(match)
                else:
                    title_candidate = match

                # Clean up the title
                title_candidate = re.sub(r'\s+', ' ', title_candidate.strip())
                candidates.append((title_candidate, 0.9))

        # Look for multi-line title patterns
        lines = [line.strip() for line in text.split('\n') if line.strip()]

        # Check for title spanning first few lines
        for i in range(min(5, len(lines))):
            line = lines[i]

            # Skip if line is too short or contains obvious non-title content
            if len(line) < 5 or any(skip in line.lower() for skip in ['seite', 'page', 'stand vom']):
                continue

            # Check if this could be part of a multi-line title
            if 'anforderungen' in line.lower():
                # Try to combine with surrounding lines
                title_parts = [line]

                # Look backwards for title parts
                for j in range(i-1, max(-1, i-3), -1):
                    if j >= 0 and lines[j] and len(lines[j]) > 3:
                        if any(word in lines[j].lower() for word in ['berufs', 'eintritt', 'lehrberuf']):
                            title_parts.insert(0, lines[j])

                # Look forwards for profession name
                for j in range(i+1, min(len(lines), i+3)):
                    if lines[j] and len(lines[j]) > 3:
                        # Check if this looks like a profession name
                        if re.match(r'^[A-ZÄÖÜ][a-zäöüß]{5,25}$', lines[j]):
                            title_parts.append(lines[j])
                            break

                if len(title_parts) > 1:
                    combined_title = ' '.join(title_parts)
                    combined_title = re.sub(r'\s+', ' ', combined_title.strip())
                    candidates.append((combined_title, 0.85))

        # Fall back to original method if no enhanced matches
        if not candidates:
            return super().extract_title(text)

        # Return best candidate
        best_title, best_score = max(candidates, key=lambda x: x[1])

        if self.debug:
            print(f"   🏆 Enhanced title: '{best_title}' (score: {best_score:.2f})")

        return best_title, best_score

    def extract_enhanced_publisher(self, text: str) -> Tuple[Optional[str], float]:
        """Enhanced publisher extraction for German institutions"""
        if self.debug:
            print("🏢 ENHANCED PUBLISHER EXTRACTION...")

        candidates = []

        # Try enhanced patterns
        for pattern in self.enhanced_publisher_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_match = re.sub(r'\s+', ' ', match.strip())
                if len(clean_match) > 5:
                    candidates.append((clean_match, 0.9))

        # Look for "bearbeitet vom" pattern (common in German documents)
        bearbeitet_pattern = r'bearbeitet\s+vom\s*\n?\s*([^.\n]{10,80})'
        matches = re.findall(bearbeitet_pattern, text, re.IGNORECASE)
        for match in matches:
            clean_match = re.sub(r'\s+', ' ', match.strip())
            candidates.append((clean_match, 0.8))

        # Look for organization patterns in first 20 lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for i, line in enumerate(lines[:20]):
            # Check for German institution keywords
            institution_indicators = [
                'ausschuss', 'ausschuß', 'kammer', 'arbeitsfront',
                'reichsgruppe', 'arbeitsgemeinschaft', 'ministerium'
            ]

            if any(indicator in line.lower() for indicator in institution_indicators):
                # Try to extract the full institution name
                if 10 <= len(line) <= 80:
                    candidates.append((line, 0.7))

        if not candidates:
            return super().extract_publisher(text)

        # Return best candidate
        best_publisher, best_score = max(candidates, key=lambda x: x[1])

        if self.debug:
            print(f"   🏆 Enhanced publisher: '{best_publisher}' (score: {best_score:.2f})")

        return best_publisher, best_score

    def extract_enhanced_document_type(self, text: str) -> Tuple[Optional[str], float]:
        """Enhanced document type extraction"""
        if self.debug:
            print("📄 ENHANCED DOCUMENT TYPE EXTRACTION...")

        candidates = []

        # Use enhanced patterns
        for doc_type, patterns in self.enhanced_doc_type_patterns.items():
            type_score = 0.0
            match_count = 0

            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                if matches > 0:
                    match_count += matches
                    type_score += matches * 0.3

            if match_count > 0:
                # Bonus for title/early appearance
                early_text = text[:1000]
                if any(re.search(pattern, early_text, re.IGNORECASE) for pattern in patterns):
                    type_score *= 1.4

                candidates.append((doc_type, min(type_score, 1.0)))

        if not candidates:
            return super().extract_document_type(text)

        # Return best candidate
        best_type, best_score = max(candidates, key=lambda x: x[1])

        if self.debug:
            print(f"   🏆 Enhanced document type: '{best_type}' (score: {best_score:.2f})")

        return best_type, best_score

    def extract_profession(self, text: str) -> Tuple[Optional[str], float]:
        """Extract specific profession/trade from the document"""
        if self.debug:
            print("⚙️ EXTRACTING PROFESSION...")

        candidates = []

        # Look for profession patterns
        for pattern in self.enhanced_profession_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                candidates.append((match, 0.8))

        # Look in first few lines for profession names
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:10]:
            # Check if line looks like a profession name
            if re.match(r'^[A-ZÄÖÜ][a-zäöüß]{5,25}$', line):
                candidates.append((line, 0.7))

        if not candidates:
            return None, 0.0

        # Return best candidate
        best_profession, best_score = max(candidates, key=lambda x: x[1])

        if self.debug:
            print(f"   🏆 Profession: '{best_profession}' (score: {best_score:.2f})")

        return best_profession, best_score

    def extract_all_metadata_enhanced(self, text: str) -> Dict:
        """Extract all metadata using enhanced methods"""
        if self.debug:
            print("🔍 ENHANCED METADATA EXTRACTION...")

        # Get original metadata
        original_metadata = super().extract_all_metadata(text)

        # Apply enhanced extractors
        enhanced_title, title_conf = self.extract_enhanced_title(text)
        enhanced_publisher, pub_conf = self.extract_enhanced_publisher(text)
        enhanced_doc_type, type_conf = self.extract_enhanced_document_type(text)
        profession, prof_conf = self.extract_profession(text)

        # Update with enhanced results if they're better
        if enhanced_title and title_conf > original_metadata.confidence_scores.get('title', 0):
            original_metadata.title = enhanced_title
            original_metadata.confidence_scores['title'] = title_conf

        if enhanced_publisher and pub_conf > original_metadata.confidence_scores.get('publisher', 0):
            original_metadata.publisher = enhanced_publisher
            original_metadata.confidence_scores['publisher'] = pub_conf

        if enhanced_doc_type and type_conf > original_metadata.confidence_scores.get('document_type', 0):
            original_metadata.document_type = enhanced_doc_type
            original_metadata.confidence_scores['document_type'] = type_conf

        # Add profession if found
        if profession:
            original_metadata.confidence_scores['profession'] = prof_conf
            # Update document type to include profession if it's more specific
            if prof_conf > 0.7:
                original_metadata.document_type = f"{enhanced_doc_type or 'Berufsanforderungen'} - {profession}"

        if self.debug:
            print("✅ ENHANCED METADATA EXTRACTION COMPLETE")
            print(f"   Enhanced Title: {original_metadata.title}")
            print(f"   Enhanced Publisher: {original_metadata.publisher}")
            print(f"   Enhanced Document Type: {original_metadata.document_type}")
            print(f"   Profession: {profession}")

        return original_metadata


