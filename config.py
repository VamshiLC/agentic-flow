"""
Configuration for ASH Infrastructure Detection Agent

Centralized configuration for models, processing parameters, and categories.
"""
import os


class Config:
    """Main configuration class for the detection agent"""

    # ===== Model Configuration =====
    # Qwen2.5-VL settings (Updated to 7B for better accuracy)
    QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
    QWEN_SERVER_URL = os.getenv("QWEN_SERVER_URL", "http://0.0.0.0:8001/v1")
    QWEN_API_KEY = os.getenv("LLM_API_KEY", "DUMMY_API_KEY")

    # SAM3 settings
    SAM3_CONFIDENCE_THRESHOLD = float(os.getenv("SAM3_CONFIDENCE", "0.5"))
    SAM3_DEVICE = "cuda"  # or "cpu"

    # ===== Processing Configuration =====
    # Video processing
    VIDEO_SAMPLE_RATE = int(os.getenv("VIDEO_SAMPLE_RATE", "15"))  # 2 frames per second at 30fps
    DEFAULT_FPS = 30

    # Output settings
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
    SAVE_JSON = True
    SAVE_MASKS = True
    DEBUG_MODE = os.getenv("DEBUG", "false").lower() == "true"

    # ===== Directory Structure =====
    MODELS_DIR = "models"
    ASSETS_DIR = "assets"
    EXAMPLES_DIR = "examples"
    OUTPUT_FRAMES_DIR = "frames"
    OUTPUT_DETECTIONS_DIR = "detections"

    # ===== Detection Categories =====
    # All 12 infrastructure categories with metadata
    CATEGORIES = {
        "potholes": {
            "name": "Pothole",
            "severity": "critical_high",
            "color": "red",
            "priority": 1,
            "description": "Severe road defects, holes in pavement surface"
        },
        "alligator_cracks": {
            "name": "Alligator Crack",
            "severity": "critical_high",
            "color": "red",
            "priority": 1,
            "description": "Web-like cracking patterns resembling alligator skin"
        },
        "abandoned_vehicles": {
            "name": "Abandoned Vehicle",
            "severity": "medium",
            "color": "yellow",
            "priority": 2,
            "description": "Derelict or abandoned vehicles on or near the road"
        },
        "longitudinal_cracks": {
            "name": "Longitudinal Crack",
            "severity": "non_critical_low",
            "color": "green",
            "priority": 3,
            "description": "Cracks running parallel to the direction of traffic"
        },
        "transverse_cracks": {
            "name": "Transverse Crack",
            "severity": "non_critical_low",
            "color": "green",
            "priority": 3,
            "description": "Cracks running perpendicular to the direction of traffic"
        },
        "damaged_paint": {
            "name": "Damaged Paint",
            "severity": "non_critical_low",
            "color": "green",
            "priority": 3,
            "description": "Deteriorated or faded road markings and painted lines"
        },
        "manholes": {
            "name": "Manhole",
            "severity": "non_critical_low",
            "color": "green",
            "priority": 3,
            "description": "Manhole covers and utility access points"
        },
        "dumped_trash": {
            "name": "Dumped Trash",
            "severity": "non_critical_low",
            "color": "green",
            "priority": 3,
            "description": "Debris, litter, or illegally dumped items"
        },
        "street_signs": {
            "name": "Street Sign",
            "severity": "non_critical_low",
            "color": "green",
            "priority": 3,
            "description": "Traffic signs, street name signs, regulatory signs"
        },
        "traffic_lights": {
            "name": "Traffic Light",
            "severity": "non_critical_low",
            "color": "green",
            "priority": 3,
            "description": "Traffic signal lights and poles"
        },
        "tyre_marks": {
            "name": "Tyre Mark",
            "severity": "non_critical_low",
            "color": "green",
            "priority": 3,
            "description": "Tire marks or skid marks on pavement"
        },
        "damaged_crosswalks": {
            "name": "Damaged Crosswalk",
            "severity": "non_critical_low",
            "color": "green",
            "priority": 3,
            "description": "Deteriorated or faded pedestrian crosswalk markings"
        }
    }

    # Category lists by severity
    CRITICAL_CATEGORIES = ["potholes", "alligator_cracks"]
    MEDIUM_CATEGORIES = ["abandoned_vehicles"]
    LOW_CATEGORIES = [
        "longitudinal_cracks", "transverse_cracks", "damaged_paint",
        "manholes", "dumped_trash", "street_signs", "traffic_lights",
        "tyre_marks", "damaged_crosswalks"
    ]

    @classmethod
    def get_category_info(cls, category):
        """Get metadata for a specific category"""
        return cls.CATEGORIES.get(category, {})

    @classmethod
    def get_all_categories(cls):
        """Get list of all category names"""
        return list(cls.CATEGORIES.keys())

    @classmethod
    def is_valid_category(cls, category):
        """Check if a category is valid"""
        return category in cls.CATEGORIES

    @classmethod
    def get_severity(cls, category):
        """Get severity level for a category"""
        return cls.CATEGORIES.get(category, {}).get("severity", "non_critical_low")

    @classmethod
    def get_color(cls, category):
        """Get color for a category"""
        return cls.CATEGORIES.get(category, {}).get("color", "green")


class SageMakerConfig(Config):
    """SageMaker-specific configuration overrides"""

    # Override paths for SageMaker environment
    OUTPUT_DIR = os.getenv("SM_OUTPUT_DATA_DIR", "/opt/ml/output")
    MODELS_DIR = os.getenv("SM_MODEL_DIR", "/opt/ml/model")

    # SageMaker resource configuration
    INSTANCE_TYPE = os.getenv("SM_CURRENT_INSTANCE_TYPE", "ml.g4dn.xlarge")
    NUM_GPUS = int(os.getenv("SM_NUM_GPUS", "1"))


# Helper function to get config based on environment
def get_config():
    """
    Get appropriate config based on environment.

    Returns:
        Config: Configuration class (SageMakerConfig if running on SageMaker, else Config)
    """
    if os.getenv("SM_CURRENT_HOST"):
        # Running on SageMaker
        return SageMakerConfig
    else:
        # Running locally
        return Config
