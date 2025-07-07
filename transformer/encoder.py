"""
Transformer Encoder Implementation

This module implements the encoder part of the transformer architecture.
The encoder consists of multiple identical layers, each containing:
1. Multi-head self-attention
2. Position-wise feed-forward network
3. Layer normalization and residual connections

The encoder processes the input sequence and produces contextual representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    
    Consists of two linear transformations with a ReLU activation in between.
    Applied to each position separately and identically.
    
    Args:
        d_model (int): Model dimension
        d_ff (int): Feed-forward dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass of position-wise feed-forward network
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            torch.Tensor: Output tensor
        """
        # First linear transformation with ReLU
        x = F.relu(self.linear1(x))
        # Apply dropout
        x = self.dropout(x)
        # Second linear transformation
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    
    A single layer of the transformer encoder containing:
    - Multi-head self-attention
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
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass of encoder layer
        
        Args:
            x (torch.Tensor): Input tensor
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Output tensor
            torch.Tensor: Attention weights (for visualization)
        """
        # Self-attention with residual connection and layer norm
        attn_output, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    
    Stack of multiple transformer encoder layers.
    Processes input sequence and produces contextual representations.
    
    Args:
        d_model (int): Model dimension
        num_layers (int): Number of encoder layers
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward dimension
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, num_layers: int, num_heads: int, 
                 d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        """
        Forward pass of transformer encoder
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Encoder output
            list: Attention weights from all layers (for visualization)
        """
        attention_weights = []
        
        # Pass through each encoder layer
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            attention_weights.append(attn_weights)
        
        # Final layer normalization
        x = self.norm(x)
        
        return x, attention_weights 