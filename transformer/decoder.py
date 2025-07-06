"""
Transformer Decoder Implementation

This module implements the decoder part of the transformer architecture.
The decoder consists of multiple identical layers, each containing:
1. Masked multi-head self-attention (causal attention)
2. Multi-head encoder-decoder attention
3. Position-wise feed-forward network
4. Layer normalization and residual connections

The decoder generates output sequences autoregressively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    
    A single layer of the transformer decoder containing:
    - Masked multi-head self-attention (causal)
    - Multi-head encoder-decoder attention
    - Position-wise feed-forward network
    - Layer normalization and residual connections
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: Add masked self-attention, encoder-decoder attention, and feed-forward
        pass
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass of decoder layer
        
        Args:
            x (torch.Tensor): Decoder input tensor
            encoder_output (torch.Tensor): Output from encoder
            src_mask (torch.Tensor, optional): Source attention mask
            tgt_mask (torch.Tensor, optional): Target attention mask (causal)
            
        Returns:
            torch.Tensor: Output tensor
            tuple: Attention weights (self-attention, encoder-attention)
        """
        # TODO: Implement decoder layer
        pass


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder
    
    Stack of multiple transformer decoder layers.
    Generates output sequences autoregressively.
    
    Args:
        d_model (int): Model dimension
        num_layers (int): Number of decoder layers
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, num_layers: int, num_heads: int, 
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        # TODO: Add decoder layers
        pass
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        Forward pass of transformer decoder
        
        Args:
            x (torch.Tensor): Decoder input tensor of shape (batch_size, seq_len, d_model)
            encoder_output (torch.Tensor): Output from encoder
            src_mask (torch.Tensor, optional): Source attention mask
            tgt_mask (torch.Tensor, optional): Target attention mask (causal)
            
        Returns:
            torch.Tensor: Decoder output
            tuple: Attention weights from all layers (for visualization)
        """
        # TODO: Implement decoder forward pass
        pass


def generate_square_subsequent_mask(size):
    """
    Generate a square mask for the sequence.
    The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    
    Args:
        size (int): Size of the mask
        
    Returns:
        torch.Tensor: Square mask of shape (size, size)
    """
    # TODO: Implement square subsequent mask
    # Upper triangular matrix with -inf values
    # This prevents the decoder from attending to future tokens
    pass


def create_causal_mask(size):
    """
    Create causal mask for autoregressive generation
    
    Args:
        size (int): Sequence length
        
    Returns:
        torch.Tensor: Causal mask
    """
    # TODO: Implement causal mask
    # Similar to square subsequent mask but with different shape
    pass 