"""
Transformer from Scratch

A complete implementation of the Transformer architecture from the ground up.
This package contains all the core components needed to build and train transformers.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main components for easy access
from .attention import MultiHeadAttention
from .encoder import TransformerEncoder, TransformerEncoderLayer
from .decoder import TransformerDecoder, TransformerDecoderLayer
from .model import Transformer

__all__ = [
    "MultiHeadAttention",
    "TransformerEncoder",
    "TransformerEncoderLayer", 
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "Transformer",
] 