"""
Exemplar Module for Few-Shot Infrastructure Detection

This module provides few-shot learning capabilities using image exemplars
to improve detection accuracy for infrastructure defects.

Components:
- ExemplarManager: Manages exemplar loading, storage, and retrieval
- ExemplarPromptBuilder: Builds multi-image prompts for Qwen3-VL
- SAM3ExemplarSegmenter: SAM3 exemplar-based segmentation
"""

from .exemplar_manager import ExemplarManager, Exemplar, CategoryExemplars
from .prompt_builder import ExemplarPromptBuilder
from .sam3_exemplar import SAM3ExemplarSegmenter

__all__ = [
    "ExemplarManager",
    "Exemplar",
    "CategoryExemplars",
    "ExemplarPromptBuilder",
    "SAM3ExemplarSegmenter"
]
