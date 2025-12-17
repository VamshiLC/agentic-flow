"""
Exemplar Manager for Few-Shot Infrastructure Detection

Manages loading, storage, and retrieval of exemplar images for few-shot
learning with both Qwen3-VL and SAM3.

Features:
- Pre-stored library exemplars (assets/exemplars/)
- Runtime CLI-uploaded exemplars
- Unlimited exemplars per category
- Support for positive and negative exemplars
"""

import os
import json
import glob
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class Exemplar:
    """Single exemplar with metadata."""
    image_path: str
    category: str
    is_positive: bool = True
    bbox: Optional[List[int]] = None  # [x1, y1, x2, y2]
    description: str = ""
    severity: Optional[str] = None
    source: str = "library"  # "library" or "runtime"
    confusion_reason: Optional[str] = None  # For negative exemplars
    _image: Optional[Image.Image] = field(default=None, repr=False)

    @property
    def image(self) -> Optional[Image.Image]:
        """Lazy load image when accessed."""
        if self._image is None and os.path.exists(self.image_path):
            try:
                self._image = Image.open(self.image_path).convert('RGB')
            except Exception as e:
                logger.error(f"Failed to load exemplar image {self.image_path}: {e}")
        return self._image

    def load_image(self) -> Optional[Image.Image]:
        """Force load the image."""
        return self.image

    def get_cropped_region(self) -> Optional[Image.Image]:
        """Get the cropped region if bbox is available."""
        if self.bbox is None or self.image is None:
            return self.image

        x1, y1, x2, y2 = self.bbox
        return self.image.crop((x1, y1, x2, y2))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "filename": os.path.basename(self.image_path),
            "bbox": self.bbox,
            "description": self.description,
            "severity": self.severity,
            "source": self.source,
            "confusion_reason": self.confusion_reason
        }


@dataclass
class CategoryExemplars:
    """All exemplars for a category."""
    category: str
    description: str = ""
    severity: str = "non_critical_low"
    positive: List[Exemplar] = field(default_factory=list)
    negative: List[Exemplar] = field(default_factory=list)

    @property
    def total_count(self) -> int:
        """Total number of exemplars."""
        return len(self.positive) + len(self.negative)

    @property
    def has_exemplars(self) -> bool:
        """Check if any exemplars exist."""
        return len(self.positive) > 0 or len(self.negative) > 0


class ExemplarManager:
    """
    Manages exemplar loading, storage, and retrieval for few-shot learning.

    Supports:
    - Pre-stored library exemplars (assets/exemplars/)
    - Runtime CLI-uploaded exemplars
    - Unlimited exemplars per category
    - Both positive and negative exemplars

    Example:
        manager = ExemplarManager("assets/exemplars")

        # Add runtime exemplar
        manager.add_runtime_exemplar("potholes", "/path/to/image.jpg", bbox=[100, 150, 300, 350])

        # Get exemplars for detection
        exemplars = manager.get_exemplars("potholes", max_count=5)

        # Prepare for Qwen3-VL
        qwen_data = manager.prepare_for_qwen("potholes", target_image)
    """

    # Valid categories
    VALID_CATEGORIES = [
        "potholes", "alligator_cracks", "longitudinal_cracks", "transverse_cracks",
        "manholes", "damaged_paint", "damaged_crosswalks", "dumped_trash",
        "street_signs", "traffic_lights", "tyre_marks", "abandoned_vehicles"
    ]

    def __init__(
        self,
        library_dir: str = "assets/exemplars",
        auto_load: bool = True
    ):
        """
        Initialize ExemplarManager.

        Args:
            library_dir: Directory containing exemplar library
            auto_load: If True, automatically load library on init
        """
        self.library_dir = library_dir
        self.metadata_file = os.path.join(library_dir, "exemplars.json")

        # Storage for all categories
        self._exemplars: Dict[str, CategoryExemplars] = {}

        # Initialize empty categories
        for category in self.VALID_CATEGORIES:
            self._exemplars[category] = CategoryExemplars(category=category)

        # Load library if requested
        if auto_load and os.path.exists(library_dir):
            self.load_library()

    def load_library(self) -> Dict[str, CategoryExemplars]:
        """
        Load all exemplars from the library directory.

        Returns:
            Dict mapping category names to CategoryExemplars
        """
        logger.info(f"Loading exemplar library from {self.library_dir}")

        # Load metadata if exists
        metadata = {}
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    metadata = data.get("categories", {})
            except Exception as e:
                logger.warning(f"Failed to load exemplars.json: {e}")

        # Process each category
        for category in self.VALID_CATEGORIES:
            category_dir = os.path.join(self.library_dir, category)

            if not os.path.exists(category_dir):
                continue

            # Get category metadata
            cat_meta = metadata.get(category, {})
            self._exemplars[category].description = cat_meta.get("description", "")
            self._exemplars[category].severity = cat_meta.get("severity", "non_critical_low")

            # Load positive exemplars
            positive_dir = os.path.join(category_dir, "positive")
            if os.path.exists(positive_dir):
                positive_meta = {e["filename"]: e for e in cat_meta.get("positive_exemplars", [])}
                self._load_exemplars_from_dir(
                    positive_dir, category, is_positive=True, metadata=positive_meta
                )

            # Load negative exemplars
            negative_dir = os.path.join(category_dir, "negative")
            if os.path.exists(negative_dir):
                negative_meta = {e["filename"]: e for e in cat_meta.get("negative_exemplars", [])}
                self._load_exemplars_from_dir(
                    negative_dir, category, is_positive=False, metadata=negative_meta
                )

            # Also check root category dir for images (backwards compatibility)
            root_meta = {e["filename"]: e for e in cat_meta.get("positive_exemplars", [])}
            self._load_exemplars_from_dir(
                category_dir, category, is_positive=True, metadata=root_meta, recursive=False
            )

        # Log stats
        stats = self.get_stats()
        logger.info(f"Loaded {stats['total_exemplars']} exemplars across {stats['categories_with_exemplars']} categories")

        return self._exemplars

    def _load_exemplars_from_dir(
        self,
        directory: str,
        category: str,
        is_positive: bool,
        metadata: Dict[str, Dict] = None,
        recursive: bool = True
    ):
        """Load exemplar images from a directory."""
        if not os.path.exists(directory):
            return

        # Find image files
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        image_files = []

        for pattern in patterns:
            if recursive:
                image_files.extend(glob.glob(os.path.join(directory, "**", pattern), recursive=True))
            else:
                image_files.extend(glob.glob(os.path.join(directory, pattern)))

        # Skip subdirectories when loading from root
        if not recursive:
            image_files = [f for f in image_files if os.path.dirname(f) == directory]

        metadata = metadata or {}

        for image_path in image_files:
            filename = os.path.basename(image_path)
            meta = metadata.get(filename, {})

            exemplar = Exemplar(
                image_path=image_path,
                category=category,
                is_positive=is_positive,
                bbox=meta.get("bbox"),
                description=meta.get("description", ""),
                severity=meta.get("severity"),
                source="library",
                confusion_reason=meta.get("confusion_reason")
            )

            if is_positive:
                # Avoid duplicates
                if not any(e.image_path == image_path for e in self._exemplars[category].positive):
                    self._exemplars[category].positive.append(exemplar)
            else:
                if not any(e.image_path == image_path for e in self._exemplars[category].negative):
                    self._exemplars[category].negative.append(exemplar)

    def add_runtime_exemplar(
        self,
        category: str,
        image_path: str,
        bbox: Optional[List[int]] = None,
        is_positive: bool = True,
        description: str = ""
    ) -> Optional[Exemplar]:
        """
        Add a runtime exemplar from CLI upload.

        Args:
            category: Category name (must be valid)
            image_path: Path to the exemplar image
            bbox: Optional bounding box [x1, y1, x2, y2]
            is_positive: True for positive exemplar, False for negative
            description: Optional description

        Returns:
            The created Exemplar, or None if invalid
        """
        # Normalize category name
        category = self._normalize_category(category)

        if category not in self.VALID_CATEGORIES:
            logger.error(f"Invalid category: {category}. Valid: {self.VALID_CATEGORIES}")
            return None

        if not os.path.exists(image_path):
            logger.error(f"Exemplar image not found: {image_path}")
            return None

        exemplar = Exemplar(
            image_path=os.path.abspath(image_path),
            category=category,
            is_positive=is_positive,
            bbox=bbox,
            description=description,
            source="runtime"
        )

        if is_positive:
            self._exemplars[category].positive.append(exemplar)
        else:
            self._exemplars[category].negative.append(exemplar)

        logger.info(f"Added runtime {'positive' if is_positive else 'negative'} exemplar for {category}: {image_path}")

        return exemplar

    def add_runtime_exemplar_dir(self, dir_path: str) -> int:
        """
        Add all exemplars from a runtime directory.

        Expected structure:
        dir_path/
        ├── potholes/
        │   ├── positive/
        │   └── negative/
        ├── cracks/
        └── ...

        Args:
            dir_path: Path to directory containing category folders

        Returns:
            Number of exemplars added
        """
        if not os.path.exists(dir_path):
            logger.error(f"Directory not found: {dir_path}")
            return 0

        count = 0

        for category in self.VALID_CATEGORIES:
            category_dir = os.path.join(dir_path, category)

            if not os.path.exists(category_dir):
                continue

            # Load positive
            positive_dir = os.path.join(category_dir, "positive")
            if os.path.exists(positive_dir):
                before = len(self._exemplars[category].positive)
                self._load_exemplars_from_dir(positive_dir, category, is_positive=True)
                # Mark as runtime
                for e in self._exemplars[category].positive[before:]:
                    e.source = "runtime"
                count += len(self._exemplars[category].positive) - before

            # Load negative
            negative_dir = os.path.join(category_dir, "negative")
            if os.path.exists(negative_dir):
                before = len(self._exemplars[category].negative)
                self._load_exemplars_from_dir(negative_dir, category, is_positive=False)
                for e in self._exemplars[category].negative[before:]:
                    e.source = "runtime"
                count += len(self._exemplars[category].negative) - before

        logger.info(f"Added {count} runtime exemplars from {dir_path}")
        return count

    def get_exemplars(
        self,
        category: str,
        include_negative: bool = False,
        max_count: Optional[int] = None,
        source_filter: Optional[str] = None
    ) -> List[Exemplar]:
        """
        Get exemplars for a specific category.

        Args:
            category: Category name
            include_negative: Include negative exemplars
            max_count: Maximum number to return (None for unlimited)
            source_filter: Filter by source ("library" or "runtime")

        Returns:
            List of Exemplar objects
        """
        category = self._normalize_category(category)

        if category not in self._exemplars:
            return []

        cat_exemplars = self._exemplars[category]
        result = list(cat_exemplars.positive)

        if include_negative:
            result.extend(cat_exemplars.negative)

        # Filter by source
        if source_filter:
            result = [e for e in result if e.source == source_filter]

        # Limit count
        if max_count is not None and len(result) > max_count:
            result = result[:max_count]

        return result

    def get_positive_exemplars(
        self,
        category: str,
        max_count: Optional[int] = None
    ) -> List[Exemplar]:
        """Get only positive exemplars for a category."""
        return self.get_exemplars(category, include_negative=False, max_count=max_count)

    def get_negative_exemplars(self, category: str) -> List[Exemplar]:
        """Get negative exemplars for a category (for false positive filtering)."""
        category = self._normalize_category(category)
        if category not in self._exemplars:
            return []
        return list(self._exemplars[category].negative)

    def has_exemplars(self, category: str) -> bool:
        """Check if a category has any exemplars."""
        category = self._normalize_category(category)
        if category not in self._exemplars:
            return False
        return self._exemplars[category].has_exemplars

    def get_all_categories(self) -> List[str]:
        """Get list of categories that have exemplars."""
        return [cat for cat in self.VALID_CATEGORIES if self.has_exemplars(cat)]

    def prepare_for_qwen(
        self,
        category: str,
        target_image: Optional[Image.Image] = None,
        max_exemplars: int = 3,
        include_negative: bool = False
    ) -> Dict[str, Any]:
        """
        Prepare exemplars for Qwen3-VL multi-image prompt.

        Args:
            category: Category to get exemplars for
            target_image: Optional target image to analyze
            max_exemplars: Maximum positive exemplars to include
            include_negative: Include negative exemplars for contrastive

        Returns:
            {
                "images": [PIL.Image, ...],  # Exemplar images + target
                "positive_descriptions": ["desc1", ...],
                "negative_descriptions": ["desc1", ...],
                "category": str,
                "has_exemplars": bool
            }
        """
        positive = self.get_positive_exemplars(category, max_count=max_exemplars)
        negative = self.get_negative_exemplars(category) if include_negative else []

        result = {
            "images": [],
            "positive_descriptions": [],
            "negative_descriptions": [],
            "category": category,
            "has_exemplars": len(positive) > 0
        }

        # Add positive exemplar images
        for exemplar in positive:
            img = exemplar.get_cropped_region() if exemplar.bbox else exemplar.image
            if img:
                result["images"].append(img)
                result["positive_descriptions"].append(
                    exemplar.description or f"Example of {category}"
                )

        # Add negative exemplar info (descriptions only, to save context)
        for exemplar in negative[:2]:  # Limit negative to 2
            result["negative_descriptions"].append(
                exemplar.confusion_reason or exemplar.description or "Not this"
            )

        # Add target image last
        if target_image:
            result["images"].append(target_image)

        return result

    def prepare_for_sam3(
        self,
        category: str,
        max_exemplars: int = 5
    ) -> Dict[str, Any]:
        """
        Prepare exemplar bboxes for SAM3 bbox-based prompting.

        Args:
            category: Category to get exemplars for
            max_exemplars: Maximum exemplars to include

        Returns:
            {
                "exemplar_images": [PIL.Image, ...],
                "bboxes": [[x1, y1, x2, y2], ...],
                "category": str,
                "has_bboxes": bool
            }
        """
        exemplars = self.get_positive_exemplars(category, max_count=max_exemplars)

        result = {
            "exemplar_images": [],
            "bboxes": [],
            "category": category,
            "has_bboxes": False
        }

        for exemplar in exemplars:
            if exemplar.image:
                result["exemplar_images"].append(exemplar.image)
                if exemplar.bbox:
                    result["bboxes"].append(exemplar.bbox)
                    result["has_bboxes"] = True

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about loaded exemplars."""
        stats = {
            "total_exemplars": 0,
            "total_positive": 0,
            "total_negative": 0,
            "categories_with_exemplars": 0,
            "library_exemplars": 0,
            "runtime_exemplars": 0,
            "by_category": {}
        }

        for category, cat_exemplars in self._exemplars.items():
            pos_count = len(cat_exemplars.positive)
            neg_count = len(cat_exemplars.negative)
            total = pos_count + neg_count

            if total > 0:
                stats["categories_with_exemplars"] += 1
                stats["total_exemplars"] += total
                stats["total_positive"] += pos_count
                stats["total_negative"] += neg_count

                # Count by source
                for e in cat_exemplars.positive + cat_exemplars.negative:
                    if e.source == "library":
                        stats["library_exemplars"] += 1
                    else:
                        stats["runtime_exemplars"] += 1

                stats["by_category"][category] = {
                    "positive": pos_count,
                    "negative": neg_count,
                    "total": total
                }

        return stats

    def save_metadata(self, output_path: Optional[str] = None):
        """
        Save current exemplar state to metadata file.

        Args:
            output_path: Path to save (defaults to library metadata file)
        """
        output_path = output_path or self.metadata_file

        data = {
            "version": "1.0",
            "description": "Exemplar library for few-shot infrastructure detection",
            "categories": {}
        }

        for category, cat_exemplars in self._exemplars.items():
            if cat_exemplars.total_count == 0:
                continue

            data["categories"][category] = {
                "description": cat_exemplars.description,
                "severity": cat_exemplars.severity,
                "positive_exemplars": [e.to_dict() for e in cat_exemplars.positive],
                "negative_exemplars": [e.to_dict() for e in cat_exemplars.negative]
            }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved exemplar metadata to {output_path}")

    def _normalize_category(self, category: str) -> str:
        """Normalize category name to match internal format."""
        # Convert to lowercase and replace spaces/hyphens with underscores
        normalized = category.lower().strip().replace(" ", "_").replace("-", "_")

        # Handle common variations
        variations = {
            "pothole": "potholes",
            "crack": "longitudinal_cracks",
            "alligator_crack": "alligator_cracks",
            "longitudinal_crack": "longitudinal_cracks",
            "transverse_crack": "transverse_cracks",
            "manhole": "manholes",
            "crosswalk": "damaged_crosswalks",
            "damaged_crosswalk": "damaged_crosswalks",
            "trash": "dumped_trash",
            "sign": "street_signs",
            "street_sign": "street_signs",
            "traffic_light": "traffic_lights",
            "tyre_mark": "tyre_marks",
            "tire_mark": "tyre_marks",
            "tire_marks": "tyre_marks",
            "abandoned_vehicle": "abandoned_vehicles",
            "vehicle": "abandoned_vehicles"
        }

        return variations.get(normalized, normalized)

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"ExemplarManager({stats['total_exemplars']} exemplars, {stats['categories_with_exemplars']} categories)"
