"""
Focused Date Extraction for German Legal Documents
Based on the specific pattern: "(Stand vom 1. Januar 1938)"
"""

import re
from datetime import datetime


class ExtractorGermanDate:
    def __init__(self, debug=False):
        self.german_months = {
            'januar': 1, 'jan': 1,
            'februar': 2, 'feb': 2,
            'märz': 3, 'mär': 3, 'maerz': 3,
            'april': 4, 'apr': 4,
            'mai': 5,
            'juni': 6, 'jun': 6,
            'juli': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'oktober': 10, 'okt': 10,
            'november': 11, 'nov': 11,
            'dezember': 12, 'dez': 12
        }

        # Add common OCR errors for month names
        self.ocr_month_corrections = {
            'zamuar': 'januar',  # Common OCR error
            'zanuar': 'januar',
            'jamuar': 'januar',
            'jänner': 'januar',
            'aprıl': 'april',
            'deeember': 'dezember',
            'dezernber': 'dezember',
        }

        self.date_patterns = [
            # Original pattern with OCR error tolerance
            r'\b(?:[Ss]tand\s+)?vom\s+(\d{1,2})\.\s*(\w+)\s+((?:19|20)\d{2})\b',

            # Handle parentheses that might be OCR'd incorrectly
            r'[({]\s*[Ss]tand\s+vom\s+(\d{1,2})\.\s*(\w+)\s+((?:19|20)\d{2})\s*[)}]',

            # Looser pattern for OCR errors
            r'vom\s+(\d{1,2})\.\s*(\w+)\s+((?:19|20)\d{2})',
        ]

        self.debug = False

    def correct_month_name(self, month_name):
        """Correct common OCR errors in month names"""
        month_lower = month_name.lower()

        # Check direct matches first
        if month_lower in self.german_months:
            return month_lower

        # Check OCR corrections
        if month_lower in self.ocr_month_corrections:
            return self.ocr_month_corrections[month_lower]

        # Try partial matching for severe OCR errors
        for correct_month in self.german_months.keys():
            if len(correct_month) > 4:  # Only for longer month names
                # Check if 70% of characters match
                matches = sum(1 for a, b in zip(month_lower, correct_month) if a == b)
                if matches >= len(correct_month) * 0.7:
                    return correct_month

        return month_lower  # Return original if no correction found


    def extract_date(self, text):
        """
        Extract date with OCR error handling
        """
        if self.debug:
            print("🗓️ Extracting date with OCR error correction...")

        for i, pattern in enumerate(self.date_patterns):
            if self.debug:
                print(f"   Trying pattern {i + 1}: {pattern}")

            matches = re.findall(pattern, text, re.IGNORECASE)

            for match in matches:
                day, month_name, year = match

                if self.debug:
                    print(f"   Found: Day={day}, Month={month_name}, Year={year}")

                # Correct month name for OCR errors
                corrected_month = self.correct_month_name(month_name)
                month_num = self.german_months.get(corrected_month)

                if self.debug and corrected_month != month_name.lower():
                    print(f"   Corrected '{month_name}' to '{corrected_month}'")

                if month_num:
                    try:
                        from datetime import datetime
                        date_obj = datetime(int(year), month_num, int(day))
                        result = date_obj.strftime("%d.%m.%Y")

                        if self.debug:
                            print(f"   ✅ Valid date found: {result}")

                        return result
                    except ValueError as e:
                        if self.debug:
                            print(f"   ❌ Invalid date: {e}")
                        continue
                else:
                    if self.debug:
                        print(f"   ❌ Unknown month: {corrected_month}")

        if self.debug:
            print("   ❌ No valid date found")
        return None
