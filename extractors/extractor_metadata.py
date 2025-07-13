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


@dataclass
class ExtractedMetadata:
    """Structured metadata container"""
    title: Optional[str] = None
    year: Optional[int] = None
    date: Optional[datetime] = None
    publisher: Optional[str] = None
    author: Optional[str] = None
    document_type: Optional[str] = None
    confidence_scores: Dict[str, float] = None
    raw_text_preview: Optional[str] = None

    def to_dict(self):
        return {
            'title': self.title,
            'year': self.year,
            'date': self.date.isoformat() if self.date else None,
            'publisher': self.publisher,
            'author': self.author,
            'document_type': self.document_type,
            'confidence_scores': self.confidence_scores or {},
            'raw_text_preview': self.raw_text_preview
        }
