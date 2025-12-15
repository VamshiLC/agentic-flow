"""
Detailed Category Prompts for Improved Detection Accuracy

Each category includes:
- Visual description
- What to look for (positive indicators)
- What it's NOT (negative examples)
- Context/location clues
"""

DETAILED_PROMPTS = {
    "potholes": {
        "what": "Bowl-shaped holes or depressions in road pavement",
        "visual_cues": [
            "Dark cavity or depression in road surface",
            "Broken, jagged, or crumbling asphalt edges",
            "May contain water, debris, or exposed base layer",
            "Typically 6+ inches in diameter with visible depth",
            "Surface damage penetrating through asphalt layer"
        ],
        "location": "On road surfaces, driving lanes, shoulders, parking lots",
        "not_this": [
            "Small surface cracks without depth",
            "Paint marks or road markings",
            "Shadows or dark patches on pavement",
            "Manhole covers or utility access",
            "Tire marks or skid marks",
            "Water puddles on intact pavement"
        ]
    },

    "alligator_cracks": {
        "what": "Web-like interconnected cracks resembling alligator skin pattern",
        "visual_cues": [
            "Network of intersecting cracks forming polygon shapes",
            "Resembles alligator or crocodile skin texture",
            "Multiple cracks connecting at various angles",
            "Typically covers area larger than 1 square meter",
            "May show slight pavement displacement between cracks"
        ],
        "location": "On road surfaces, especially high-traffic areas",
        "not_this": [
            "Single linear cracks",
            "Random scattered small cracks",
            "Expansion joints (straight regular patterns)",
            "Surface texture or road grain",
            "Shadows from tree branches"
        ]
    },

    "longitudinal_cracks": {
        "what": "Cracks running parallel to the direction of traffic/road centerline",
        "visual_cues": [
            "Linear crack running along traffic direction",
            "Follows road length, not width",
            "May be continuous or intermittent",
            "Typically appears near lane edges or wheel paths",
            "Can be thin line or wider gap"
        ],
        "location": "Along road length, often near lane markings",
        "not_this": [
            "Transverse cracks (across the road)",
            "Lane markings or paint lines",
            "Expansion joints",
            "Shadows from overhead objects"
        ]
    },

    "transverse_cracks": {
        "what": "Cracks running perpendicular to traffic direction, across the road width",
        "visual_cues": [
            "Linear crack crossing road width",
            "Runs perpendicular to centerline",
            "Often straight or slightly curved",
            "May span full lane width or partial",
            "Can indicate temperature/structural stress"
        ],
        "location": "Across road width, perpendicular to traffic flow",
        "not_this": [
            "Longitudinal cracks (along road)",
            "Crosswalk markings",
            "Speed bumps or road features",
            "Shadows from buildings/trees"
        ]
    },

    "graffiti": {
        "what": "Unauthorized spray paint, tags, or markings on public infrastructure",
        "visual_cues": [
            "MUST have CLEAR, VISIBLE spray-painted text, symbols, or artwork",
            "Letters, words, tags, or artistic designs painted on surfaces",
            "Often bright colors (especially red, blue, black, white, yellow)",
            "Applied to walls, barriers, signs, bridges, or overpasses",
            "May include gang tags, street art, or vandalism markings",
            "Distinct from official signage or road markings",
            "IMPORTANT: Must be CLEARLY VISIBLE paint/spray on surface"
        ],
        "location": "Walls, bridges, underpasses, barriers, utility boxes, signs, buildings",
        "not_this": [
            "⚠️ NEVER detect shadows, stains, or natural discoloration",
            "⚠️ NEVER detect dirt, rust, or weathering",
            "⚠️ NEVER detect official road markings or signage",
            "⚠️ NEVER detect poster advertisements or stickers",
            "Construction markings or utility paint",
            "Faded paint on infrastructure (not graffiti)",
            "Natural patterns or textures on walls",
            "Reflections or lighting effects",
            "CRITICAL: If it's not clearly painted text/art, DO NOT DETECT!"
        ]
    },

    "abandoned_vehicle": {
        "what": "Severely damaged, deteriorated vehicle or car being used as shelter (homeless situation)",
        "visual_cues": [
            "CRITICAL: Must have MULTIPLE (3+) of these indicators:",
            "- Flat, missing, or severely deflated tires",
            "- Broken, shattered, or missing windows/windshield",
            "- Thick layer of dust, dirt covering entire vehicle (not just light dust)",
            "- Covered with tarps, blankets, cardboard, or paper (homeless shelter)",
            "- Person visibly living inside with belongings/bedding",
            "- Vegetation/weeds growing around wheels or under vehicle",
            "- Extensive rust, body damage, or missing parts (bumpers, doors, hood)",
            "- Faded paint with peeling/chipping covering large areas",
            "- Missing or expired license plates",
            "- Appears unmoved for extended period (debris accumulation)",
            "ONLY detect if vehicle looks CLEARLY non-functional or used as shelter"
        ],
        "location": "Streets, alleys, vacant lots, under bridges - NOT parking lots with clean cars",
        "not_this": [
            "⚠️ NEVER detect regular parked vehicles - even if stationary",
            "⚠️ Clean vehicles in parking lots or driveways",
            "⚠️ Cars with shiny paint or clear windows",
            "⚠️ Vehicles with inflated tires and no visible damage",
            "⚠️ Commercial vehicles, delivery trucks, work vans",
            "⚠️ Cars that look functional and roadworthy",
            "⚠️ Temporarily parked cars (even if dusty from weather)",
            "⚠️ Vehicles in organized parking areas",
            "⚠️ Cars with visible license plates and registration",
            "IMPORTANT: If vehicle looks like it could drive away, DO NOT detect it!"
        ]
    },

    "homeless_encampment": {
        "what": "ACTUAL makeshift shelter with VISIBLE tents, tarps, or people living in public space",
        "visual_cues": [
            "CRITICAL: Must have CLEAR, VISIBLE evidence of ALL of these:",
            "- Actual tents (fabric dome structures) OR tarps suspended/draped as shelter",
            "- Visible sleeping bags, blankets, or bedding on the ground",
            "- Personal belongings, bags, or shopping carts clearly visible",
            "- Human presence or signs of active habitation",
            "IMPORTANT: Vegetation, weeds, trash alone are NOT encampments",
            "MUST see actual shelter structures (tents/tarps) to qualify"
        ],
        "location": "Under bridges, in doorways, vacant lots, parks, sidewalks - WITH visible shelters",
        "not_this": [
            "⚠️ NEVER detect vegetation, weeds, or overgrown areas",
            "⚠️ NEVER detect empty fenced areas with plants",
            "⚠️ NEVER detect trash or debris without shelter structures",
            "⚠️ NEVER detect dirt patches or bare ground",
            "Camping tents in designated campgrounds",
            "Construction worker break areas",
            "Organized outdoor events",
            "Stored equipment or materials",
            "Temporary vendor setups",
            "Random piles of items without visible shelter",
            "CRITICAL: If you don't see tents or tarps, DO NOT DETECT!"
        ]
    },

    "homeless_person": {
        "what": "ACTUAL PERSON visible with clear indicators of street homelessness",
        "visual_cues": [
            "CRITICAL: Must have VISIBLE HUMAN PERSON + homelessness indicators:",
            "MUST see actual person's body, head, or clear human form",
            "Person lying/sleeping on ground with visible bedding/belongings",
            "Individual with shopping cart FULL of possessions",
            "Person with makeshift cardboard shelter actively present",
            "Person panhandling with visible sign",
            "Person sitting/sleeping on sidewalk with multiple bags",
            "IMPORTANT: MUST see the ACTUAL PERSON - not just belongings!",
            "ONLY detect if you can clearly identify a human being present"
        ],
        "location": "Sidewalks, parks, under bridges, doorways",
        "not_this": [
            "⚠️ NEVER detect empty areas, vegetation, or shadows",
            "⚠️ NEVER detect belongings/objects without visible person",
            "⚠️ NEVER detect fences, barriers, or construction materials",
            "⚠️ NEVER detect trash, debris, or random items",
            "⚠️ NEVER detect tents/shelters WITHOUT seeing the person",
            "Regular pedestrians walking or standing",
            "Shoppers, workers, or tourists",
            "People at bus stops or waiting areas",
            "Runners, cyclists, or people exercising",
            "CRITICAL: If you can't clearly see a HUMAN PERSON, DO NOT DETECT!"
        ]
    },

    "manholes": {
        "what": "Circular or rectangular utility access covers in pavement",
        "visual_cues": [
            "Circular or square metal cover",
            "Often has text, patterns, or municipality markings",
            "Flush with or slightly below pavement level",
            "May have pick holes or lifting points",
            "Distinct from surrounding pavement material",
            "Usually dark metal (iron/steel) appearance"
        ],
        "location": "In roads, sidewalks, intersections",
        "not_this": [
            "Storm drain grates (linear openings)",
            "Potholes or pavement damage",
            "Painted circles on pavement",
            "Shadows or dark spots",
            "Vehicle covers or caps"
        ]
    },

    "damaged_paint": {
        "what": "Deteriorated, faded, or worn road markings and painted lines ON PAVEMENT ONLY",
        "visual_cues": [
            "MUST be painted directly ON the asphalt/concrete road surface",
            "Faded white or yellow lane lines or arrows on road",
            "Partially worn lane divider markings on pavement",
            "Chipped or peeling paint showing black asphalt underneath",
            "Barely visible road arrows, text, or symbols on pavement",
            "Ghosted or shadow images of old markings on road surface",
            "Uneven or patchy paint application on pavement"
        ],
        "location": "Road surfaces, crosswalks, parking lots - ONLY ON GROUND/PAVEMENT",
        "not_this": [
            "⚠️ NEVER detect painted fences, barriers, or construction equipment",
            "⚠️ NEVER detect yellow/white objects that are NOT paint on road",
            "⚠️ NEVER detect curbs, poles, or vertical structures",
            "Fresh, clear road markings",
            "Intentionally removed markings",
            "Shadows on painted lines",
            "Dirt or debris on otherwise intact paint",
            "Painted signs or boards (not on road surface)",
            "Colored fencing or barriers"
        ]
    },

    "damaged_crosswalks": {
        "what": "Faded, worn, or deteriorated pedestrian crosswalk markings",
        "visual_cues": [
            "Faded white zebra stripes or ladder pattern",
            "Partially visible crosswalk bars",
            "Worn paint showing pavement underneath",
            "Incomplete or missing sections of crosswalk",
            "Discolored or ghost images of crosswalk lines"
        ],
        "location": "Intersections, mid-block crossings, school zones",
        "not_this": [
            "Fresh, clearly visible crosswalks",
            "Brick or raised crosswalks (intentional design)",
            "Wet crosswalks appearing dark",
            "Shadows crossing the road"
        ]
    },

    "dumped_trash": {
        "what": "Illegally dumped debris, garbage, or discarded items in public areas",
        "visual_cues": [
            "Piles of garbage bags or loose trash",
            "Discarded furniture, mattresses, or appliances",
            "Construction debris or waste materials",
            "Scattered litter accumulation",
            "Abandoned household items in public space",
            "Clearly not in designated trash receptacles"
        ],
        "location": "Sidewalks, alleys, vacant lots, roadsides",
        "not_this": [
            "Trash bins or dumpsters",
            "Organized recycling areas",
            "Construction materials in active work zones",
            "Yard waste for scheduled pickup",
            "Organized outdoor storage"
        ]
    },

    "street_signs": {
        "what": "Traffic signs, regulatory signs, or street name signs",
        "visual_cues": [
            "Mounted on poles or posts",
            "Reflective surface with text or symbols",
            "Standard shapes (octagon, triangle, rectangle, circle)",
            "Standard colors (red, yellow, green, blue, white)",
            "Official traffic control devices",
            "Street name signs, stop signs, yield signs, speed limit signs"
        ],
        "location": "Roadsides, intersections, above roadways",
        "not_this": [
            "Billboards or advertisements",
            "Building signs or storefronts",
            "Construction site signs (unless official traffic signs)",
            "Private property signs",
            "Decorative signage"
        ]
    },

    "traffic_lights": {
        "what": "Traffic signal lights and poles controlling vehicle/pedestrian flow",
        "visual_cues": [
            "Vertical or horizontal arrangement of lights",
            "Red, yellow, and green lights (circular or arrows)",
            "Mounted on poles or overhead gantries",
            "Pedestrian signal lights (walk/don't walk)",
            "Signal control boxes at intersections",
            "Multiple lights for different lanes/directions"
        ],
        "location": "Intersections, pedestrian crossings, above roadways",
        "not_this": [
            "Street lights or lamp posts",
            "Building lights or decorations",
            "Reflectors or reflective markers",
            "Construction warning lights",
            "Vehicle lights or headlights"
        ]
    },

    "tyre_marks": {
        "what": "DISTINCT tire skid marks or rubber deposits on pavement from hard braking/burnouts",
        "visual_cues": [
            "CRITICAL: Must be CLEARLY VISIBLE black rubber streaks",
            "MUST show parallel lines matching tire tread pattern",
            "Dark black rubber deposits distinctly darker than pavement",
            "Skid marks from hard braking showing curve or straight line",
            "Burnout marks (very dark, concentrated rubber)",
            "MUST be fresh and clearly visible - not faded or ambiguous",
            "IMPORTANT: If not OBVIOUSLY tire marks, DO NOT detect"
        ],
        "location": "Road surfaces at intersections, turns, or accident sites",
        "not_this": [
            "⚠️ NEVER detect painted road markings or lines",
            "⚠️ NEVER detect oil stains or fluid spills",
            "⚠️ NEVER detect shadows from vehicles or trees",
            "⚠️ NEVER detect pavement cracks, joints, or seams",
            "⚠️ NEVER detect dirt, mud, or water tracks",
            "⚠️ NEVER detect normal pavement discoloration",
            "Faded or barely visible marks",
            "Shadows or dark patches on road",
            "Road texture or aggregate patterns",
            "CRITICAL: Only detect if marks are CLEARLY from tires!"
        ]
    }
}


def build_detailed_prompt(categories):
    """Build comprehensive detection prompt with detailed descriptions."""

    prompt = """You are an EXPERT infrastructure inspector analyzing street imagery with HIGH PRECISION.

Your task is to ACCURATELY detect infrastructure issues and objects. Use the detailed criteria below.

"""

    for i, category in enumerate(categories, 1):
        if category not in DETAILED_PROMPTS:
            continue

        details = DETAILED_PROMPTS[category]

        prompt += f"\n{'='*70}\n"
        prompt += f"{i}. {category.upper().replace('_', ' ')}\n"
        prompt += f"{'='*70}\n\n"

        prompt += f"WHAT IT IS:\n{details['what']}\n\n"

        prompt += f"VISUAL INDICATORS:\n"
        for cue in details['visual_cues']:
            prompt += f"  ✓ {cue}\n"
        prompt += "\n"

        prompt += f"TYPICAL LOCATION:\n  {details['location']}\n\n"

        prompt += f"⚠️ DO NOT CONFUSE WITH:\n"
        for neg in details['not_this']:
            prompt += f"  ✗ {neg}\n"
        prompt += "\n"

    prompt += f"\n{'='*70}\n"
    prompt += """
OUTPUT FORMAT - RESPOND EXACTLY AS SHOWN:

For each detection found:
Defect: <category_name>, Box: [x1, y1, x2, y2], Confidence: <0.0-1.0>

Where:
- category_name = EXACT category name from above (use underscores)
- Coordinates = 0-1000 scale (normalized to image)
- x1,y1 = top-left corner
- x2,y2 = bottom-right corner
- Confidence = your confidence level (0.0 to 1.0)

EXAMPLES:
Defect: potholes, Box: [120, 450, 380, 620], Confidence: 0.92
Defect: homeless_person, Box: [50, 100, 300, 800], Confidence: 0.85
Defect: abandoned_vehicle, Box: [200, 150, 900, 550], Confidence: 0.78

If NO defects match the criteria: "No defects detected"

🚨 CRITICAL ACCURACY RULES - READ CAREFULLY 🚨:

⚠️ FALSE POSITIVES ARE EXTREMELY COSTLY - BE VERY CONSERVATIVE! ⚠️

✓ ONLY detect if you are ABSOLUTELY CERTAIN (>85%) it matches ALL criteria
✓ Read the "DO NOT CONFUSE WITH" section CAREFULLY - these are common errors!
✓ If ANYTHING looks ambiguous, unclear, or questionable → DO NOT DETECT!
✓ It is MUCH BETTER to miss a detection than to create a false positive
✓ For sensitive categories (homeless_person, homeless_encampment, abandoned_vehicle, tyre_marks):
  - Require MULTIPLE (3+) clear visual indicators - not just one!
  - NEVER detect based on shadows, vegetation, fences, or unclear objects
  - MUST see the actual object clearly - not just assume it's there
  - Err on the side of NOT detecting rather than guessing

✓ Give realistic confidence scores - NEVER inflate them
✓ Double-check EVERY detection against ALL the negative examples
✓ Question yourself: "Could this be something else?" If yes → DO NOT DETECT!

🚫 ABSOLUTE "NEVER DETECT" RULES:
• Shadows, reflections, or lighting effects → NEVER homeless_person/encampment
• Vegetation, weeds, or plants → NEVER homeless_encampment
• Empty fenced areas or barriers → NEVER homeless_person/encampment
• Regular parked cars → NEVER abandoned_vehicle
• Road discoloration or shadows → NEVER tyre_marks
• If you can't clearly identify the object → DO NOT DETECT ANY CATEGORY!

MANDATORY ACCURACY CHECKLIST FOR EACH DETECTION:
1. Does it match ALL (not just some) of the visual cues? YES/NO
2. Is it explicitly NOT in the "DO NOT CONFUSE WITH" list? YES/NO
3. Does the location make perfect sense for this category? YES/NO
4. Am I >85% confident this is correct? YES/NO
5. Have I eliminated ALL other possibilities? YES/NO
6. Could this possibly be a shadow, vegetation, or normal object? NO required
7. Would a human expert agree with this detection? YES/NO

If you answer NO to questions 1-5, 7 OR YES to question 6 → DO NOT DETECT!

WHEN IN DOUBT → DO NOT DETECT! Zero detections is better than false positives!

Now carefully analyze this image:
"""

    return prompt
