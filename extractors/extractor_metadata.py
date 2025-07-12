
"""
Modular Metadata Extraction - One function per entity
Start with easy entities first, then train AI models progressively
"""
import re

from extractors.extractor_german_date import ExtractorGermanDate


class ExtractorMetadata:
    def __init__(self, debug=False):
        self.trained_models = {}
        self.debug = debug


    # =================== EASY ENTITIES (START HERE) ===================

    def extract_date(self, text):
        """
        Extract full date from text in German legal format: 'DD.MM.YYYY'
        """
        if self.debug:
            print("📅 Extracting DATE...")

        german_date_extractor = ExtractorGermanDate()
        german_date_extractor.debug = self.debug

        extracted_date = german_date_extractor.extract_date(text)

        return extracted_date  # Already in 'DD.MM.YYYY' format or None
