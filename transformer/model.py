"""
Complete Transformer Model Implementation

This module implements the complete transformer architecture combining:
1. Input/Output embeddings
2. Positional encoding
3. Encoder stack
4. Decoder stack
5. Final output projection

The transformer can be used for sequence-to-sequence tasks like machine translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .encoder import TransformerEncoder
from .decoder import TransformerDecoder, generate_square_subsequent_mask


class PositionalEncoding(nn.Module):
    """
    Positional Encoding
    
    Adds position information to the input embeddings since the transformer
    has no recurrence or convolution. Uses sine and cosine functions of
    different frequencies.
    
    Args:
        d_model (int): Model dimension
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input embeddings
        
        Args:
            x (torch.Tensor): Input embeddings of shape (seq_len, batch_size, d_model)
            
        Returns:
            torch.Tensor: Embeddings with positional encoding added
        """
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:seq_len, :].transpose(0, 1)
        return self.dropout(x)


class Transformer(nn.Module):
    """
    Complete Transformer Model
    
    Implements the full transformer architecture as described in "Attention Is All You Need".
    Combines encoder and decoder with embeddings and positional encoding.
    
    Args:
        src_vocab_size (int): Source vocabulary size
        tgt_vocab_size (int): Target vocabulary size
        d_model (int): Model dimension
        num_layers (int): Number of encoder/decoder layers
        num_heads (int): Number of attention heads
        d_ff (int): Feed-forward dimension
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 num_layers: int = 6, num_heads: int = 8, d_ff: int = 2048,
                 max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder and decoder
        self.encoder = TransformerEncoder(d_model, num_layers, num_heads, d_ff, dropout)
        self.decoder = TransformerDecoder(d_model, num_layers, num_heads, d_ff, dropout)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize model weights
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass of the transformer
        
        Args:
            src (torch.Tensor): Source sequence
            tgt (torch.Tensor): Target sequence
            src_mask (torch.Tensor, optional): Source attention mask
            tgt_mask (torch.Tensor, optional): Target attention mask
            
        Returns:
            torch.Tensor: Output logits
        """
        # Encode source sequence
        encoder_output, _ = self.encode(src, src_mask)
        
        # Decode target sequence
        decoder_output, _ = self.decode(tgt, encoder_output, src_mask, tgt_mask)
        
        # Project to vocabulary size
        output = self.output_projection(decoder_output)
        
        return output
    
    def encode(self, src, src_mask=None):
        """
        Encode source sequence
        
        Args:
            src (torch.Tensor): Source sequence
            src_mask (torch.Tensor, optional): Source attention mask
            
        Returns:
            torch.Tensor: Encoder output
            list: Encoder attention weights
        """
        # Source embeddings + positional encoding
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        
        # Pass through encoder
        encoder_output, attention_weights = self.encoder(src_emb, src_mask)
        
        return encoder_output, attention_weights
    
    def decode(self, tgt, encoder_output, src_mask=None, tgt_mask=None):
        """
        Decode target sequence given encoder output
        
        Args:
            tgt (torch.Tensor): Target sequence
            encoder_output (torch.Tensor): Output from encoder
            src_mask (torch.Tensor, optional): Source attention mask
            tgt_mask (torch.Tensor, optional): Target attention mask
            
        Returns:
            torch.Tensor: Decoder output
            tuple: Decoder attention weights
        """
        # Target embeddings + positional encoding
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        # Pass through decoder
        decoder_output, attention_weights = self.decoder(
            tgt_emb, encoder_output, src_mask, tgt_mask
        )
        
        return decoder_output, attention_weights
    
    def generate(self, src, src_mask=None, max_len=50, start_token=1, end_token=2):
        """
        Generate target sequence autoregressively
        
        Args:
            src (torch.Tensor): Source sequence
            src_mask (torch.Tensor, optional): Source attention mask
            max_len (int): Maximum generation length
            start_token (int): Start token ID
            end_token (int): End token ID
            
        Returns:
            torch.Tensor: Generated sequence
        """
        self.eval()
        
        # Encode source sequence
        encoder_output, _ = self.encode(src, src_mask)
        
        # Initialize target sequence with start token
        batch_size = src.size(0)
        device = src.device
        tgt = torch.tensor([[start_token]] * batch_size, device=device)
        
        # Generate tokens one by one
        for _ in range(max_len):
            # Create causal mask for current target sequence
            tgt_len = tgt.size(1)
            tgt_mask = generate_square_subsequent_mask(tgt_len).to(device)
            
            # Expand mask to match batch size if needed
            if tgt_mask.dim() == 2:
                tgt_mask = tgt_mask.unsqueeze(0).expand(batch_size, -1, -1)
            
            # Decode current target sequence
            decoder_output, _ = self.decode(tgt, encoder_output, src_mask, tgt_mask)
            
            # Get next token logits (only the last position)
            next_token_logits = self.output_projection(decoder_output[:, -1, :])
            
            # Sample next token (greedy decoding)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to target sequence
            tgt = torch.cat([tgt, next_token], dim=1)
            
            # Check if all sequences have generated end token
            if (next_token == end_token).all():
                break
        
        return tgt


def create_transformer_model(src_vocab_size, tgt_vocab_size, **kwargs):
    """
    Factory function to create a transformer model with default parameters
    
    Args:
        src_vocab_size (int): Source vocabulary size
        tgt_vocab_size (int): Target vocabulary size
        **kwargs: Additional arguments to override defaults
        
    Returns:
        Transformer: Configured transformer model
    """
    # Default parameters (from the original paper)
    defaults = {
        'd_model': 512,
        'num_layers': 6,
        'num_heads': 8,
        'd_ff': 2048,
        'max_len': 5000,
        'dropout': 0.1
    }
    
    # Update with provided arguments
    defaults.update(kwargs)
    
    return Transformer(src_vocab_size, tgt_vocab_size, **defaults) 