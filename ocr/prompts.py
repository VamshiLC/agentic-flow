"""
License Plate Detection and OCR Prompts

Specialized prompts for:
1. Detecting vehicles with visible license plates
2. Extracting text from cropped plate images
3. Focused on North American formats (US, Canada, Mexico)
"""


def build_plate_detection_prompt() -> str:
    """
    Build prompt for detecting license plates on vehicles.
    Returns bounding boxes in 0-1000 normalized coordinates.
    """
    return """You are a vehicle license plate detection expert specializing in North American plates.

YOUR TASK: Detect ONLY actual vehicle license/number plates in this image.

WHAT IS A LICENSE PLATE:
- RECTANGULAR metal or plastic plate (NOT round or circular)
- Contains ALPHANUMERIC characters (letters AND/OR numbers)
- Has a FLAT surface with printed/embossed text
- Usually has a state/province name at top or bottom
- Standard size: approximately 12"x6" (wider than tall)
- Colors: white, yellow, or light colored background with dark text
- Mounted FLAT on vehicle bumper (front or rear)

PLATE LOCATIONS:
- Front bumper center area
- Rear bumper/trunk center area
- Mounted horizontally (not tilted or vertical)

CRITICAL - DO NOT DETECT THESE AS PLATES:
- TYRES/WHEELS (round, black rubber - NOT a plate!)
- Hubcaps or wheel rims (circular metal - NOT a plate!)
- Headlights or taillights (lights - NOT a plate!)
- Bumper stickers or decals
- Vehicle manufacturer logos/badges
- Dealer name plates or frames
- Exhaust pipes or grilles
- Side mirrors
- Door handles
- ANY circular or round objects
- ANY black rubber objects
- Building numbers or street signs

DETECTION RULES:
1. ONLY detect rectangular plates with visible text
2. Plate must have alphanumeric characters visible
3. Ignore anything that is round, circular, or wheel-shaped
4. Ignore anything that is black rubber (tyres)
5. Minimum confidence 0.6 for detection
6. If unsure, do NOT detect it

OUTPUT FORMAT:
For each license plate detected:
Defect: license_plate, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>

Where coordinates are in 0-1000 scale (normalized to image dimensions).

EXAMPLES:
Defect: license_plate, Box: [450, 680, 550, 720], Confidence: 0.95
Defect: license_plate, Box: [120, 400, 280, 450], Confidence: 0.82

If NO license plates are visible: "No defects detected"

Now analyze this image for license plates ONLY (not tyres, wheels, or other parts):"""


def build_ocr_prompt() -> str:
    """
    Build prompt for extracting text from a cropped license plate image.
    Optimized for North American plate formats with confusion handling.
    """
    return """You are an expert license plate OCR specialist. Read the plate text CHARACTER BY CHARACTER.

YOUR TASK: Extract the EXACT text from this license plate image.

CRITICAL - AVOID COMMON OCR MISTAKES:
These characters are often confused - look carefully:
- 0 (zero) vs O (letter O) - zero is more oval, O is rounder
- 1 (one) vs I (letter I) vs L (letter L) - check for serifs
- 8 (eight) vs B (letter B) - 8 has two equal loops, B has unequal bumps
- 5 (five) vs S (letter S) - 5 has flat top, S is curved
- 2 (two) vs Z (letter Z) - 2 has curved bottom, Z is angular
- 6 (six) vs G (letter G) - 6 is closed, G has horizontal bar
- 4 (four) vs A (letter A) - 4 is angular, A has crossbar lower

READING RULES:
1. Read LEFT to RIGHT, one character at a time
2. License plates use ONLY UPPERCASE letters (A-Z) and digits (0-9)
3. Look for spaces or hyphens between character groups
4. Focus on the MAIN plate text, ignore small state slogans
5. If a character is unclear, make your best guess based on context

NORTH AMERICAN FORMATS (for context):
- California: 1ABC234 (digit + 3 letters + 3 digits)
- Texas/Standard US: ABC-1234 or ABC 1234 (3 letters + 4 digits)
- Ontario: ABCD 123 (4 letters + 3 digits)
- Quebec: 123 ABC (3 digits + 3 letters)
- Mexico: ABC-12-34 (3 letters + 2 digits + 2 digits)

OUTPUT FORMAT (use exactly this format):
PlateText: <THE_EXACT_CHARACTERS>
Confidence: <0.0-1.0>
State: <state/province if visible, else "Unknown">
Format: <brief format description>

EXAMPLES:

PlateText: 7ABC123
Confidence: 0.95
State: California
Format: California standard

PlateText: ABC-1234
Confidence: 0.92
State: Texas
Format: US standard

PlateText: ABCD 123
Confidence: 0.88
State: Ontario
Format: Ontario standard

If completely unreadable:
PlateText: UNREADABLE
Confidence: 0.0
State: Unknown
Format: Unknown

Now carefully read each character on this plate:"""


def build_combined_detection_ocr_prompt() -> str:
    """
    Build prompt that does detection + OCR in one pass.
    Use for simpler/faster processing when high accuracy isn't critical.
    """
    return """You are a vehicle license plate detection and OCR expert for North American plates.

YOUR TASK:
1. Detect ALL visible vehicle license plates in this image
2. For each plate, extract the text/numbers if readable

NORTH AMERICAN PLATES TO DETECT:
- US state plates (all 50 states)
- Canadian province plates
- Mexican state plates

OUTPUT FORMAT:
For each plate found:
Plate: Box: [x1, y1, x2, y2], Text: <PLATE_TEXT>, Confidence: <0.0-1.0>, State: <STATE_NAME>

Where:
- Coordinates are in 0-1000 scale (normalized)
- Text is the exact characters on the plate (or "UNREADABLE" if unclear)
- Confidence reflects detection + OCR certainty combined
- State is the state/province name if visible

EXAMPLES:
Plate: Box: [450, 680, 550, 720], Text: 8ABC123, Confidence: 0.92, State: California
Plate: Box: [120, 400, 280, 450], Text: ABC-1234, Confidence: 0.85, State: Texas
Plate: Box: [300, 500, 400, 540], Text: UNREADABLE, Confidence: 0.60, State: Unknown

If NO plates visible: "No plates detected"

Now analyze this image:"""


# North American plate format patterns for validation
PLATE_PATTERNS = {
    # US States
    "california": r"^[0-9][A-Z]{3}[0-9]{3}$",  # 8ABC123
    "texas": r"^[A-Z]{3}-?[0-9]{4}$",  # ABC-1234 or ABC1234
    "new_york": r"^[A-Z]{3}-?[0-9]{4}$",  # ABC-1234
    "florida": r"^[A-Z]{3}\s?[A-Z][0-9]{2}$",  # ABC A12
    "us_standard": r"^[A-Z]{3}[\s-]?[0-9]{4}$",  # ABC 1234
    "us_vanity": r"^[A-Z0-9]{1,7}$",  # Up to 7 chars

    # Canada
    "ontario": r"^[A-Z]{4}\s?[0-9]{3}$",  # ABCD 123
    "quebec": r"^[0-9]{3}\s?[A-Z]{3}$",  # 123 ABC
    "british_columbia": r"^[A-Z]{2}[0-9]\s?[0-9]{2}[A-Z]$",  # AB1 23C
    "alberta": r"^[A-Z]{3}-?[0-9]{4}$",  # ABC-1234

    # Mexico
    "mexico": r"^[A-Z]{3}-[0-9]{2}-[0-9]{2}$",  # ABC-12-34
}

# State/Province identifiers that may appear on plates
STATE_IDENTIFIERS = {
    # US States
    "california": ["CALIFORNIA", "CA", "THE GOLDEN STATE"],
    "texas": ["TEXAS", "TX", "THE LONE STAR STATE"],
    "florida": ["FLORIDA", "FL", "SUNSHINE STATE"],
    "new_york": ["NEW YORK", "NY", "EMPIRE STATE"],
    "illinois": ["ILLINOIS", "IL", "LAND OF LINCOLN"],
    "pennsylvania": ["PENNSYLVANIA", "PA", "KEYSTONE STATE"],
    "ohio": ["OHIO", "OH", "BIRTHPLACE OF AVIATION"],
    "georgia": ["GEORGIA", "GA", "PEACH STATE"],
    "michigan": ["MICHIGAN", "MI", "GREAT LAKES"],
    "arizona": ["ARIZONA", "AZ", "GRAND CANYON STATE"],

    # Canadian Provinces
    "ontario": ["ONTARIO", "ON", "YOURS TO DISCOVER"],
    "quebec": ["QUEBEC", "QC", "JE ME SOUVIENS"],
    "british_columbia": ["BRITISH COLUMBIA", "BC", "BEAUTIFUL BC"],
    "alberta": ["ALBERTA", "AB", "WILD ROSE COUNTRY"],
    "manitoba": ["MANITOBA", "MB", "FRIENDLY MANITOBA"],
    "saskatchewan": ["SASKATCHEWAN", "SK", "LAND OF LIVING SKIES"],

    # Mexico
    "mexico": ["MEXICO", "MEX", "ESTADOS UNIDOS MEXICANOS"],
}
