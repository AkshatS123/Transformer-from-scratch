"""
Experiments and Applications

This package contains real-world applications and experiments using the transformer:
- Machine translation experiments
- Text classification tasks
- Text generation and language modeling
- Attention visualization and analysis
"""

from .translation import TranslationExperiment
from .classification import ClassificationExperiment
from .generation import GenerationExperiment

__all__ = [
    "TranslationExperiment",
    "ClassificationExperiment", 
    "GenerationExperiment",
] 