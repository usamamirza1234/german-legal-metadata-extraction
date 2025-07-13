"""
Quick Test Script - Run this to test the enhanced extractors on your data
"""
from extractors.enhanced_extractor_metadata import EnhancedMetadataExtractor

# Your OCR text (replace this with your actual extracted text)
your_ocr_text = """
--- Page 1 (OCR) ---
Berufs-
Eignungsanforderungen

für den Eintritt in den Lehrberuf

Schmuckstemfasser
bearbeitet vom

Deutschen Ausschuß für Technisches Schulweer (Datfch) E.V.
· Berlin NW7

im Einvernehmen
mit der
Deutschen Arbeitsfront
der

Neichsgruppe Industrie

. und der

Arbeitsgemeinschaft der Industrie- und Handelskammern
in der Neichswirtfchaftskammer

(Stcmd vom 22. September 1938)

Verlag von B.G.Teubner in Leipzig und Berlin

--- Page 2 (OCR) ---
Berufseignungsanforderungen

fiir den Eintritt in den Lehrberuf

Schmucksieinfasser

(Die Begriindungen für die einzelnen Eignungtsanforderungen sind aus den
Berufsanforderungen abgeleitet, wie sie sich aus dem Arbeitsgebiet, der Berufs-
ausiibung sowie den besonderen Arbeitsbedingungen des Facharbeiters ergeben.)
"""


def quick_test_enhanced_extraction():
    """Quick test of enhanced extraction on your data"""

    # Import your existing extractor


    print("🔍 TESTING ENHANCED METADATA EXTRACTION")
    print("=" * 80)

    # Test with enhanced patterns
    extractor = EnhancedMetadataExtractor(debug=True)

    # Enhanced title extraction
    print("\n📝 ENHANCED TITLE EXTRACTION:")
    title, title_conf = extract_enhanced_title(your_ocr_text)
    print(f"   Result: '{title}' (confidence: {title_conf:.2f})")

    # Enhanced publisher extraction
    print("\n🏢 ENHANCED PUBLISHER EXTRACTION:")
    publisher, pub_conf = extract_enhanced_publisher(your_ocr_text)
    print(f"   Result: '{publisher}' (confidence: {pub_conf:.2f})")

    # Enhanced document type
    print("\n📄 ENHANCED DOCUMENT TYPE:")
    doc_type, type_conf = extract_enhanced_document_type(your_ocr_text)
    print(f"   Result: '{doc_type}' (confidence: {type_conf:.2f})")

    print("\n" + "=" * 80)
    print("EXPECTED IMPROVEMENTS:")
    print("=" * 80)
    print("✅ Title should now be: 'Berufs-Eignungsanforderungen für den Eintritt in den Lehrberuf Schmucksteinfasser'")
    print("✅ Publisher should be: 'Deutschen Ausschuß für Technisches Schulwesen (Datsch) E.V.'")
    print("✅ Document type should be: 'Eignungsanforderungen - Schmucksteinfasser'")


def extract_enhanced_title(text):
    """Enhanced title extraction for multi-line titles"""
    import re

    # Pattern for multi-line title with profession
    pattern1 = r'(Berufs-?\s*\n?\s*Eignungsanforderungen)\s*\n.*?\n.*?(für den Eintritt in den Lehrberuf)\s*\n.*?\n.*?([A-ZÄÖÜ][a-zäöüß]{8,25})'

    match = re.search(pattern1, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if match:
        title = f"{match.group(1)} {match.group(2)} {match.group(3)}"
        title = re.sub(r'\s+', ' ', title.strip())
        return title, 0.95

    # Fallback: look for title components
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    title_parts = []
    for i, line in enumerate(lines[:10]):
        if 'eignungsanforderungen' in line.lower():
            # Found main title component
            title_parts.append(line)

            # Look for "für den Eintritt" in nearby lines
            for j in range(i + 1, min(len(lines), i + 5)):
                if 'eintritt' in lines[j].lower() and 'lehrberuf' in lines[j].lower():
                    title_parts.append(lines[j])

                    # Look for profession name
                    for k in range(j + 1, min(len(lines), j + 5)):
                        if re.match(r'^[A-ZÄÖÜ][a-zäöüß]{5,25}$', lines[k]):
                            title_parts.append(lines[k])
                            break
                    break
            break

    if title_parts:
        combined_title = ' '.join(title_parts)
        combined_title = re.sub(r'\s+', ' ', combined_title.strip())
        return combined_title, 0.85

    return "Eignungsanforderungen", 0.6


def extract_enhanced_publisher(text):
    """Enhanced publisher extraction"""
    import re

    # Look for "bearbeitet vom" pattern
    pattern1 = r'bearbeitet\s+vom\s*\n?\s*([^\n.]{10,80})'
    match = re.search(pattern1, text, re.IGNORECASE)
    if match:
        publisher = re.sub(r'\s+', ' ', match.group(1).strip())
        return publisher, 0.9

    # Look for institution patterns
    patterns = [
        r'(Deutschen\s+Ausschuß\s+für\s+[^\n.]{5,40})\s*\([^)]+\)\s*E\.V\.',
        r'(Deutsche\s+Arbeitsfront)',
        r'(Reichsgruppe\s+[A-ZÄÖÜ][a-zäöüß]+)',
        r'(Arbeitsgemeinschaft\s+[^\n.]{10,50})',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[0], 0.8

    return None, 0.0


def extract_enhanced_document_type(text):
    """Enhanced document type extraction"""
    import re

    # Check for specific patterns
    if re.search(r'eignungsanforderungen', text, re.IGNORECASE):
        # Check if profession is mentioned
        profession_match = re.search(r'([A-ZÄÖÜ][a-zäöüß]{8,25}fasser)', text)
        if profession_match:
            return f"Eignungsanforderungen - {profession_match.group(1)}", 0.9
        else:
            return "Eignungsanforderungen", 0.8

    return "Berufsanforderungen", 0.6


if __name__ == "__main__":
    quick_test_enhanced_extraction()