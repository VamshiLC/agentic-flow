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

    "road_surface_damage": {
        "what": "General pavement deterioration, raveling, or surface distress",
        "visual_cues": [
            "Loose aggregate or gravel on surface",
            "Rough, deteriorated pavement texture",
            "Surface raveling (aggregate loss)",
            "Faded or worn surface appearance",
            "Patchy areas with different surface quality"
        ],
        "location": "Road surfaces, especially high-traffic or aged areas",
        "not_this": [
            "Intentional textured pavement",
            "Clean gravel roads",
            "Temporary construction surfaces",
            "Dirt or dust on otherwise intact pavement"
        ]
    },

    "abandoned_vehicle": {
        "what": "Car, truck, or vehicle that appears inoperable or permanently left",
        "visual_cues": [
            "Flat or missing tires",
            "Broken or missing windows",
            "Vegetation growth around/on vehicle",
            "Heavy dust accumulation or faded paint",
            "Visible rust, damage, or decay",
            "Missing license plates or registration stickers",
            "Doors/hood left open or missing parts"
        ],
        "location": "Parked on streets, alleys, vacant lots",
        "not_this": [
            "Regularly parked vehicles",
            "Cars actively in use",
            "Delivery or commercial vehicles",
            "Temporarily parked clean vehicles",
            "Vehicles with people nearby/inside"
        ]
    },

    "homeless_encampment": {
        "what": "Makeshift shelter or living area in public space",
        "visual_cues": [
            "Tents, tarps, or makeshift shelters",
            "Multiple sleeping bags or bedding visible",
            "Shopping carts or accumulated belongings",
            "Temporary structures with personal items",
            "Cardboard, plastic, or fabric shelters",
            "Cooking equipment or personal belongings visible"
        ],
        "location": "Under bridges, in doorways, vacant lots, parks, sidewalks",
        "not_this": [
            "Camping tents in designated campgrounds",
            "Construction worker break areas",
            "Organized outdoor events",
            "Stored equipment or materials",
            "Temporary vendor setups"
        ]
    },

    "homeless_person": {
        "what": "Person living on streets with visible indicators of homelessness",
        "visual_cues": [
            "Person with sleeping bag, blankets, or bedding in public",
            "Individual with shopping cart full of belongings",
            "Person sleeping/lying on sidewalk or bench with possessions",
            "Visible makeshift shelter or cardboard housing",
            "Person panhandling with sign or cup",
            "Individual with multiple bags/belongings in non-residential area"
        ],
        "location": "Sidewalks, parks, under bridges, doorways, bus stops",
        "not_this": [
            "Regular pedestrians walking",
            "Shoppers with shopping bags",
            "Workers or delivery people",
            "Tourists with backpacks",
            "People waiting at bus stops",
            "Runners or cyclists resting"
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
        "what": "Deteriorated, faded, or worn road markings and painted lines",
        "visual_cues": [
            "Faded white or yellow road lines",
            "Partially worn lane markings",
            "Chipped or peeling paint on pavement",
            "Barely visible crosswalk markings",
            "Ghosted or shadow images of old markings",
            "Uneven or patchy paint application"
        ],
        "location": "Road surfaces, crosswalks, parking lots",
        "not_this": [
            "Fresh, clear road markings",
            "Intentionally removed markings",
            "Shadows on painted lines",
            "Dirt or debris on otherwise intact paint"
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
        "what": "Tire tracks, skid marks, or rubber deposits on pavement",
        "visual_cues": [
            "Dark rubber streaks on road surface",
            "Parallel lines showing tire tread pattern",
            "Skid marks from braking (straight or curved)",
            "Burnout marks (darker, concentrated)",
            "May show acceleration or turning patterns",
            "Black or dark brown marks on lighter pavement"
        ],
        "location": "Road surfaces, especially at turns, intersections, or accident sites",
        "not_this": [
            "Painted road markings",
            "Oil or fluid stains",
            "Shadows from vehicles or objects",
            "Pavement cracks or joints",
            "Dirt or mud tracks"
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

IMPORTANT RULES:
✓ Only detect if you are CONFIDENT (>70%) it matches the criteria
✓ Use the visual indicators and negative examples to guide you
✓ Consider location/context clues
✓ If uncertain, DO NOT detect - false positives are costly
✓ Provide realistic confidence scores based on match quality

Now carefully analyze this image:
"""

    return prompt
