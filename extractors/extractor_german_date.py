"""
German Date Extraction for Historical Documents
Extracts dates from German legal/historical texts with OCR errors
"""

import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional


class ExtractorGermanDate:
    def __init__(self, debug=False):
        self.debug = debug

        # German month names mapping
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

        # OCR corrections
        self.ocr_month_corrections = {
            'zamuar': 'januar',
            'zanuar': 'januar',
            'jamuar': 'januar',
            'jänner': 'januar',
            'aprıl': 'april',
            'deeember': 'dezember',
            'dezernber': 'dezember',
        }

        # Date pattern
        self.date_pattern = re.compile(
            r'(\d{1,2})\.\s*([a-zA-ZäöüÄÖÜß]+)\s*(\d{4})',
            re.IGNORECASE
        )

        # Publishing date indicators (patterns that suggest official date)
        self.publishing_indicators = [
            # Strong indicators (high priority)
            r'(?:Stand\s+vom|Stcmd\s+vom)',  # "Stand vom" or OCR error "Stcmd vom"
            r'(?:Berlin,?\s+den)',  # "Berlin, den 27. April 1939"
            r'(?:München,?\s+den)',  # "München, den ..."
            r'(?:Hamburg,?\s+den)',  # "Hamburg, den ..."
            r'(?:Datum\s*:)',  # "Datum: ..."
            r'(?:Ausgegeben\s+am)',  # "Ausgegeben am ..."
            r'(?:Verkündet\s+am)',  # "Verkündet am ..."

            # Medium indicators
            r'(?:vom\s+\d{1,2}\.\s*[a-zA-ZäöüÄÖÜß]+\s+\d{4})',  # "vom DD.MM.YYYY"
            r'(?:den\s+\d{1,2}\.\s*[a-zA-ZäöüÄÖÜß]+\s+\d{4})',  # "den DD.MM.YYYY"
        ]

        # Content date indicators (suggest mentioned, not publishing date)
        self.content_indicators = [
            r'(?:Erlaß|Erlass).*vom',  # "mit dem Erlaß ... vom ..."
            r'(?:Wirkung\s+vom)',  # "mit Wirkung vom ..."
            r'(?:Kraft\s+vom)',  # "in Kraft vom ..."
            r'(?:treten.*vom)',  # "treten ... vom ..."
            r'(?:seit\s+dem)',  # "seit dem ..."
            r'(?:ab\s+dem)',  # "ab dem ..."
            r'(?:erfolgte)',  # "im August 1936 erfolgten ..."
        ]

    def normalize_month(self, month_str: str) -> Optional[str]:
        """Normalize month name, handling OCR errors"""
        month_lower = month_str.lower().strip()

        if month_lower in self.ocr_month_corrections:
            month_lower = self.ocr_month_corrections[month_lower]

        if month_lower in self.german_months:
            return month_lower

        return None

    def extract_all_dates(self, text: str) -> List[Dict]:
        """Extract all dates with context information"""
        found_dates = []

        for match in self.date_pattern.finditer(text):
            day_str, month_str, year_str = match.groups()
            start_pos = match.start()
            end_pos = match.end()

            # Normalize month
            normalized_month = self.normalize_month(month_str)
            if not normalized_month:
                continue

            try:
                day = int(day_str)
                year = int(year_str)
                month_num = self.german_months[normalized_month]

                date_obj = datetime(year, month_num, day)
                original_text = match.group(0)

                # Extract context around the date (±100 characters)
                context_start = max(0, start_pos - 100)
                context_end = min(len(text), end_pos + 100)
                context = text[context_start:context_end]

                date_info = {
                    'date': date_obj,
                    'original_text': original_text,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'context': context,
                    'day': day,
                    'month': month_num,
                    'year': year,
                    'publishing_score': 0,  # Will be calculated
                    'content_score': 0  # Will be calculated
                }

                found_dates.append(date_info)

            except (ValueError, KeyError):
                continue

        return found_dates

    def score_dates(self, dates: List[Dict], text: str) -> List[Dict]:
        """Score each date based on likelihood of being publishing date"""

        for date_info in dates:
            context = date_info['context'].lower()
            position = date_info['start_pos']
            text_length = len(text)

            publishing_score = 0
            content_score = 0

            # 1. Check for publishing indicators
            for indicator in self.publishing_indicators:
                if re.search(indicator, context, re.IGNORECASE):
                    if 'stand' in indicator.lower() or 'stcmd' in indicator.lower():
                        publishing_score += 10  # Very strong indicator
                    elif 'berlin.*den' in indicator.lower():
                        publishing_score += 8  # Strong city + date pattern
                    else:
                        publishing_score += 5  # Good indicator

                    if self.debug:
                        print(f"   Publishing indicator found: {indicator} (+{publishing_score})")

            # 2. Check for content indicators (negative for publishing date)
            for indicator in self.content_indicators:
                if re.search(indicator, context, re.IGNORECASE):
                    content_score += 5
                    if self.debug:
                        print(f"   Content indicator found: {indicator} (+{content_score})")

            # 3. Position scoring (publishing dates often at beginning or end)
            relative_pos = position / text_length
            if relative_pos < 0.3:  # First 30% of document
                publishing_score += 3
            elif relative_pos > 0.7:  # Last 30% of document
                publishing_score += 4
            else:  # Middle of document (less likely to be publishing date)
                content_score += 2

            # 4. Parentheses bonus (like your "(Stand vom ...)" pattern)
            if '(' in date_info['context'] and ')' in date_info['context']:
                publishing_score += 6
                if self.debug:
                    print(f"   Parentheses bonus: +6")

            # 5. Year reasonableness (for historical docs)
            year = date_info['year']
            if 1920 <= year <= 1950:  # Historical German legal docs
                publishing_score += 2
            elif 1950 <= year <= 2025:  # Modern docs
                publishing_score += 1

            # Final scores
            date_info['publishing_score'] = publishing_score
            date_info['content_score'] = content_score
            date_info['final_score'] = publishing_score - content_score

            if self.debug:
                print(f"   Date {date_info['original_text']}: "
                      f"pub={publishing_score}, content={content_score}, "
                      f"final={date_info['final_score']}")

        return dates

    def find_publishing_date(self, text: str) -> Optional[datetime]:
        """Find the most likely publishing date"""
        if self.debug:
            print("🗓️ FINDING PUBLISHING DATE...")

        # Extract all dates
        all_dates = self.extract_all_dates(text)

        if not all_dates:
            if self.debug:
                print("   ❌ No dates found")
            return None

        if self.debug:
            print(f"   📅 Found {len(all_dates)} dates total")

        # If only one date, it's probably the publishing date
        if len(all_dates) == 1:
            if self.debug:
                print("   ✅ Only one date found - using as publishing date")
            return all_dates[0]['date']

        # Score all dates
        scored_dates = self.score_dates(all_dates, text)

        # Sort by final score (highest first)
        sorted_dates = sorted(scored_dates, key=lambda x: x['final_score'], reverse=True)

        if self.debug:
            print("   📊 Scored dates:")
            for i, date_info in enumerate(sorted_dates):
                print(f"      {i + 1}. {date_info['original_text']} "
                      f"(score: {date_info['final_score']}) "
                      f"- {date_info['context'][:50]}...")

        # Return the highest scoring date
        best_date = sorted_dates[0]

        if self.debug:
            print(f"   🏆 Best publishing date: {best_date['original_text']} "
                  f"(score: {best_date['final_score']})")

        return best_date['date']

    def find_publishing_date_with_details(self, text: str) -> Optional[Dict]:
        """Find publishing date with detailed information"""
        all_dates = self.extract_all_dates(text)

        if not all_dates:
            return None

        if len(all_dates) == 1:
            return {
                'date': all_dates[0]['date'],
                'original_text': all_dates[0]['original_text'],
                'confidence': 'high',
                'reason': 'only_date_found',
                'all_dates': all_dates
            }

        scored_dates = self.score_dates(all_dates, text)
        sorted_dates = sorted(scored_dates, key=lambda x: x['final_score'], reverse=True)

        best_date = sorted_dates[0]

        # Determine confidence
        if best_date['final_score'] >= 8:
            confidence = 'high'
        elif best_date['final_score'] >= 4:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'date': best_date['date'],
            'original_text': best_date['original_text'],
            'confidence': confidence,
            'score': best_date['final_score'],
            'context': best_date['context'],
            'all_dates': all_dates
        }

