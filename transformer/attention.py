"""
Multi-Head Attention Implementation

This module implements the core attention mechanisms used in transformers:
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Attention masking for different use cases

Key Concepts:
- Query (Q): What we're looking for
- Key (K): What we're matching against  
- Value (V): What we're retrieving
- Attention Score: How much to focus on each position
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    
    Implements the attention mechanism as described in "Attention Is All You Need":
    Attention(Q,K,V) = softmax(QK^T/√d_k)V
    
    Args:
        d_k (int): Dimension of the key vectors
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of scaled dot-product attention
        
        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_k)
            K (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_k) 
            V (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_v)
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Attention output
            torch.Tensor: Attention weights for visualization
        """
        # TODO: Implement scaled dot-product attention
        pass


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    
    Allows the model to jointly attend to information from different representation
    subspaces at different positions.
    
    Args:
        d_model (int): Model dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # TODO: Add linear projections and attention mechanism
        pass
        
    def forward(self, Q, K, V, mask=None):
        """
        Forward pass of multi-head attention
        
        Args:
            Q (torch.Tensor): Query tensor
            K (torch.Tensor): Key tensor
            V (torch.Tensor): Value tensor
            mask (torch.Tensor, optional): Attention mask
            
        Returns:
            torch.Tensor: Multi-head attention output
            torch.Tensor: Attention weights
        """
        batch_size = Q.size(0)
        
        # TODO: Implement multi-head attention
        pass


def create_padding_mask(seq, pad_idx=0):
    """
    Create padding mask for attention
    
    Args:
        seq (torch.Tensor): Input sequence
        pad_idx (int): Padding token index
        
    Returns:
        torch.Tensor: Padding mask
    """
    # TODO: Implement padding mask
    # Mask should be True for padding tokens, False for real tokens
    pass


def create_look_ahead_mask(size):
    """
    Create look-ahead mask for decoder self-attention
    
    Args:
        size (int): Sequence length
        
    Returns:
        torch.Tensor: Look-ahead mask
    """
    # TODO: Implement look-ahead mask
    # Upper triangular matrix to prevent attending to future tokens
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
    # Similar to look-ahead mask but with different shape
    pass 