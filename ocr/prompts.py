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

YOUR TASK: Detect ALL visible vehicle license/number plates in this image.

WHAT TO LOOK FOR:
- Rectangular plates on the front or rear of vehicles
- North American plate characteristics:
  * US plates: Typically 12"x6", state name at top, alphanumeric characters
  * Canadian plates: Similar size, province name, alphanumeric format
  * Mexican plates: State abbreviation, alphanumeric with hyphens
- Plates with letters, numbers, or combination
- Standard government-issued registration plates

PLATE LOCATIONS:
- Front bumper area of vehicles
- Rear bumper/trunk area of vehicles
- Mounted on brackets or recessed in bumper

DETECTION RULES:
1. Only detect clearly visible plates where characters might be readable
2. Include plates that are partially visible if significant portion shows
3. Detect both front and rear plates if visible
4. Include plates at various angles if characters are potentially distinguishable
5. Minimum confidence 0.5 for detection

DO NOT DETECT:
- Dealer logos or temporary paper plates (unless clearly readable)
- Bumper stickers or decorative plates
- Vehicle manufacturer badges
- Building numbers or street signs
- Plates that are too blurry or too far away to read

OUTPUT FORMAT:
For each license plate detected:
Defect: license_plate, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>

Where coordinates are in 0-1000 scale (normalized to image dimensions).
x1, y1 = top-left corner
x2, y2 = bottom-right corner

EXAMPLES:
Defect: license_plate, Box: [450, 680, 550, 720], Confidence: 0.95
Defect: license_plate, Box: [120, 400, 280, 450], Confidence: 0.82
Defect: license_plate, Box: [700, 550, 850, 600], Confidence: 0.78

If NO license plates are visible: "No defects detected"

Now analyze this image for license plates:"""


def build_ocr_prompt() -> str:
    """
    Build prompt for extracting text from a cropped license plate image.
    Optimized for North American plate formats.
    """
    return """You are a license plate OCR specialist for North American plates.

YOUR TASK: Read and extract ALL text from this license plate image.

NORTH AMERICAN PLATE FORMATS:

US Plates (by state examples):
- Standard: ABC 1234, ABC-1234
- California: 8ABC123 (digit + 3 letters + 3 digits)
- Texas: ABC-1234
- New York: ABC-1234
- Florida: ABC A12
- Vanity plates: Up to 7 characters, custom text

Canadian Plates:
- Ontario: ABCD 123 (4 letters + space + 3 digits)
- Quebec: 123 ABC
- British Columbia: AB1 23C
- Alberta: ABC-1234

Mexican Plates:
- Format: ABC-12-34 (3 letters + 2 digits + 2 digits)
- State abbreviation may be present

READING INSTRUCTIONS:
1. Read ALL characters on the plate (letters and numbers)
2. Maintain the exact order as they appear left to right
3. Include spaces, hyphens, or dots if present between character groups
4. Note the state/province name if visible at top or bottom
5. Ignore decorative elements, slogans, or small text
6. If characters are partially obscured, use [?] for unclear characters

OUTPUT FORMAT:
PlateText: <EXACT_TEXT_ON_PLATE>
Confidence: <0.0-1.0>
State: <state/province name if visible, otherwise "Unknown">
Format: <plate format description>

EXAMPLES:

Example 1 - California plate:
PlateText: 8ABC123
Confidence: 0.95
State: California
Format: California standard (1 digit + 3 letters + 3 digits)

Example 2 - Texas plate:
PlateText: ABC-1234
Confidence: 0.92
State: Texas
Format: US standard with hyphen

Example 3 - Ontario plate:
PlateText: ABCD 123
Confidence: 0.88
State: Ontario
Format: Ontario standard (4 letters + 3 digits)

Example 4 - Partial read:
PlateText: AB[?] 1234
Confidence: 0.65
State: Unknown
Format: US standard (partial)

If text is completely unreadable:
PlateText: UNREADABLE
Confidence: 0.0
State: Unknown
Format: Unknown

Now read the text on this license plate:"""


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
