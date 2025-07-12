
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
        self.german_date_extractor = ExtractorGermanDate(self.debug)


    # =================== EASY ENTITIES (START HERE) ===================

    def extract_date(self, text):
        """
        Extract full date from text in German legal format: 'DD.MM.YYYY'
        """
        if self.debug:
            print("📅 Extracting DATE...")

        extracted_date = self.german_date_extractor.find_publishing_date_with_details(text)

        return extracted_date  # Already in 'DD.MM.YYYY' format or None