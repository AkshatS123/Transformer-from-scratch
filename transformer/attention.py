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

Mathematical Formula:
Attention(Q,K,V) = softmax(QK^T/√d_k)V

Why √d_k scaling? Prevents softmax from entering regions with small gradients.
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
        # Step 1: Compute attention scores: Q * K^T
        scores = torch.matmul(Q, K.transpose(-2, -1))
        
        # Step 2: Scale by √d_k (prevents softmax saturation)
        scores = scores / math.sqrt(self.d_k)
        
        # Step 3: Apply mask if provided (for padding/causal attention)
        if mask is not None:
            # Handle different mask formats
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask == 0, -1e9)
            else:
                # For -inf masks, add them to scores
                scores = scores + mask
        
        # Step 4: Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 5: Apply dropout for regularization
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Multiply with values to get output
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


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
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
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
        
        # Step 1: Linear projections for Q, K, V
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Step 2: Split into multiple heads (reshape)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Adjust mask for multiple heads if provided
        if mask is not None:
            # Get the actual sequence lengths from Q and K
            q_seq_len = Q.size(2)
            k_seq_len = K.size(2)
            
            # Handle different mask input shapes
            if mask.dim() == 2:
                # Shape: (seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, -1, -1)
            elif mask.dim() == 3:
                # Shape: (batch_size, seq_len, seq_len) -> (batch_size, num_heads, seq_len, seq_len)
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            elif mask.dim() == 4:
                # Already correct shape: (batch_size, num_heads, seq_len, seq_len)
                pass
            else:
                raise ValueError(f"Unexpected mask dimension: {mask.dim()}")
            
            # Handle cross-attention case where mask shape doesn't match attention scores
            if mask.size(-2) != q_seq_len or mask.size(-1) != k_seq_len:
                # For cross-attention, we need to adapt the mask
                # If mask is (batch_size, num_heads, src_seq_len, src_seq_len) but we need (batch_size, num_heads, tgt_seq_len, src_seq_len)
                if mask.size(-2) == k_seq_len and mask.size(-1) == k_seq_len:
                    # This is a source mask for cross-attention - take the key masking part
                    # Create a mask that masks the same key positions for all queries
                    mask = mask[:, :, :1, :].expand(-1, -1, q_seq_len, -1)
        
        # Step 3: Apply scaled dot-product attention to each head
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Step 4: Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Step 5: Final linear projection
        output = self.W_o(attention_output)
        
        return output, attention_weights


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
    # Upper triangular matrix to prevent attending to future tokens
    pass 