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

from .attention import MultiHeadAttention
from .encoder import PositionwiseFeedForward


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
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.encoder_decoder_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
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
        # Masked self-attention with residual connection and layer norm
        self_attn_output, self_attn_weights = self.masked_self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Encoder-decoder attention with residual connection and layer norm
        enc_dec_attn_output, enc_dec_attn_weights = self.encoder_decoder_attention(
            x, encoder_output, encoder_output, src_mask
        )
        x = self.norm2(x + self.dropout(enc_dec_attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x, (self_attn_weights, enc_dec_attn_weights)


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
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
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
        self_attention_weights = []
        enc_dec_attention_weights = []
        
        # Pass through each decoder layer
        for layer in self.layers:
            x, (self_attn_weights, enc_dec_attn_weights) = layer(
                x, encoder_output, src_mask, tgt_mask
            )
            self_attention_weights.append(self_attn_weights)
            enc_dec_attention_weights.append(enc_dec_attn_weights)
        
        # Final layer normalization
        x = self.norm(x)
        
        return x, (self_attention_weights, enc_dec_attention_weights)


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
    # Create upper triangular matrix with -inf values
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def create_causal_mask(size):
    """
    Create causal mask for autoregressive generation
    
    Args:
        size (int): Sequence length
        
    Returns:
        torch.Tensor: Causal mask
    """
    # Create lower triangular matrix (opposite of square subsequent mask)
    mask = torch.tril(torch.ones(size, size))
    return mask 