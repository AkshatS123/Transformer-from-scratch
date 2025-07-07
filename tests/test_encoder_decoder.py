"""
Tests for Encoder-Decoder Architecture

This module contains comprehensive tests for the transformer encoder-decoder:
- Encoder functionality and shapes
- Decoder functionality and shapes  
- Complete transformer model
- Forward pass integration
- Generation functionality
"""

import torch
import torch.nn as nn
import pytest
import numpy as np
import sys
import os

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from transformer.encoder import TransformerEncoder, TransformerEncoderLayer, PositionwiseFeedForward
from transformer.decoder import TransformerDecoder, TransformerDecoderLayer, generate_square_subsequent_mask
from transformer.model import Transformer, PositionalEncoding, create_transformer_model


class TestEncoder:
    """Test suite for transformer encoder"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.batch_size = 2
        self.seq_len = 8
        self.d_model = 128
        self.num_heads = 8
        self.d_ff = 512
        self.num_layers = 2
        self.dropout = 0.1
        
        # Create test input
        self.input_tensor = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
    def test_positionwise_feedforward(self):
        """Test position-wise feed-forward network"""
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        
        output = ff(self.input_tensor)
        
        # Check output shape
        assert output.shape == self.input_tensor.shape
        
        # Check that it's not identity
        assert not torch.equal(output, self.input_tensor)
        
    def test_encoder_layer(self):
        """Test single encoder layer"""
        layer = TransformerEncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
        
        output, attn_weights = layer(self.input_tensor)
        
        # Check output shape
        assert output.shape == self.input_tensor.shape
        
        # Check attention weights exist
        assert attn_weights is not None
        
    def test_encoder_stack(self):
        """Test encoder stack"""
        encoder = TransformerEncoder(self.d_model, self.num_layers, self.num_heads, self.d_ff, self.dropout)
        
        output, attn_weights = encoder(self.input_tensor)
        
        # Check output shape
        assert output.shape == self.input_tensor.shape
        
        # Check attention weights for all layers
        assert len(attn_weights) == self.num_layers
        
    def test_encoder_with_mask(self):
        """Test encoder with attention mask"""
        encoder = TransformerEncoder(self.d_model, self.num_layers, self.num_heads, self.d_ff, self.dropout)
        
        # Create padding mask
        mask = torch.zeros(self.batch_size, self.seq_len, self.seq_len)
        mask[:, :, -2:] = float('-inf')  # Mask last 2 positions
        
        output, attn_weights = encoder(self.input_tensor, mask)
        
        # Check output shape
        assert output.shape == self.input_tensor.shape


class TestDecoder:
    """Test suite for transformer decoder"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.batch_size = 2
        self.seq_len = 6
        self.d_model = 128
        self.num_heads = 8
        self.d_ff = 512
        self.num_layers = 2
        self.dropout = 0.1
        
        # Create test inputs
        self.decoder_input = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.encoder_output = torch.randn(self.batch_size, self.seq_len, self.d_model)
        
    def test_causal_mask_generation(self):
        """Test causal mask generation"""
        size = 5
        mask = generate_square_subsequent_mask(size)
        
        # Check shape
        assert mask.shape == (size, size)
        
        # Check that it's upper triangular with -inf
        for i in range(size):
            for j in range(size):
                if j > i:
                    assert mask[i, j] == float('-inf')
                else:
                    assert mask[i, j] == 0.0
        
    def test_decoder_layer(self):
        """Test single decoder layer"""
        layer = TransformerDecoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout)
        
        output, (self_attn, enc_dec_attn) = layer(self.decoder_input, self.encoder_output)
        
        # Check output shape
        assert output.shape == self.decoder_input.shape
        
        # Check attention weights exist
        assert self_attn is not None
        assert enc_dec_attn is not None
        
    def test_decoder_stack(self):
        """Test decoder stack"""
        decoder = TransformerDecoder(self.d_model, self.num_layers, self.num_heads, self.d_ff, self.dropout)
        
        output, (self_attn_weights, enc_dec_attn_weights) = decoder(self.decoder_input, self.encoder_output)
        
        # Check output shape
        assert output.shape == self.decoder_input.shape
        
        # Check attention weights for all layers
        assert len(self_attn_weights) == self.num_layers
        assert len(enc_dec_attn_weights) == self.num_layers
        
    def test_decoder_with_causal_mask(self):
        """Test decoder with causal mask"""
        decoder = TransformerDecoder(self.d_model, self.num_layers, self.num_heads, self.d_ff, self.dropout)
        
        # Create causal mask
        tgt_mask = generate_square_subsequent_mask(self.seq_len)
        
        output, _ = decoder(self.decoder_input, self.encoder_output, tgt_mask=tgt_mask)
        
        # Check output shape
        assert output.shape == self.decoder_input.shape


class TestCompleteTransformer:
    """Test suite for complete transformer model"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.batch_size = 2
        self.src_seq_len = 10
        self.tgt_seq_len = 8
        self.src_vocab_size = 1000
        self.tgt_vocab_size = 1000
        self.d_model = 128
        self.num_heads = 8
        self.d_ff = 512
        self.num_layers = 2
        self.dropout = 0.1
        
        # Create test inputs (token indices)
        self.src_tokens = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_seq_len))
        self.tgt_tokens = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.tgt_seq_len))
        
        # Create transformer model
        self.transformer = Transformer(
            self.src_vocab_size, self.tgt_vocab_size, self.d_model, 
            self.num_layers, self.num_heads, self.d_ff, dropout=self.dropout
        )
        
    def test_positional_encoding(self):
        """Test positional encoding"""
        pe = PositionalEncoding(self.d_model, max_len=100)
        
        # Create test input
        x = torch.randn(self.batch_size, self.src_seq_len, self.d_model)
        
        output = pe(x)
        
        # Check output shape
        assert output.shape == x.shape
        
        # Check that it's not identity (positional encoding was added)
        assert not torch.equal(output, x)
        
    def test_transformer_forward_pass(self):
        """Test complete transformer forward pass"""
        output = self.transformer(self.src_tokens, self.tgt_tokens)
        
        # Check output shape: (batch_size, tgt_seq_len, tgt_vocab_size)
        expected_shape = (self.batch_size, self.tgt_seq_len, self.tgt_vocab_size)
        assert output.shape == expected_shape
        
        # Check that output contains logits (not probabilities)
        assert not torch.all((output >= 0) & (output <= 1))
        
    def test_transformer_encode_decode(self):
        """Test separate encode and decode functionality"""
        # Test encoding
        encoder_output, enc_attn = self.transformer.encode(self.src_tokens)
        
        # Check encoder output shape
        expected_enc_shape = (self.batch_size, self.src_seq_len, self.d_model)
        assert encoder_output.shape == expected_enc_shape
        
        # Check attention weights exist
        assert len(enc_attn) == self.num_layers
        
        # Test decoding
        decoder_output, dec_attn = self.transformer.decode(self.tgt_tokens, encoder_output)
        
        # Check decoder output shape
        expected_dec_shape = (self.batch_size, self.tgt_seq_len, self.d_model)
        assert decoder_output.shape == expected_dec_shape
        
        # Check attention weights exist
        assert len(dec_attn[0]) == self.num_layers  # self-attention
        assert len(dec_attn[1]) == self.num_layers  # encoder-decoder attention
        
    def test_transformer_generation(self):
        """Test autoregressive generation"""
        # Test generation with small max_len for speed
        generated = self.transformer.generate(self.src_tokens, max_len=5, start_token=1, end_token=2)
        
        # Check output shape: (batch_size, generated_seq_len)
        assert generated.shape[0] == self.batch_size
        assert generated.shape[1] <= 5 + 1  # max_len + start_token
        
        # Check that all sequences start with start_token
        assert torch.all(generated[:, 0] == 1)
        
    def test_transformer_with_masks(self):
        """Test transformer with attention masks"""
        # Create source mask (padding)
        src_mask = torch.zeros(self.batch_size, self.src_seq_len, self.src_seq_len)
        src_mask[:, :, -2:] = float('-inf')  # Mask last 2 positions
        
        # Create target mask (causal)
        tgt_mask = generate_square_subsequent_mask(self.tgt_seq_len)
        
        output = self.transformer(self.src_tokens, self.tgt_tokens, src_mask, tgt_mask)
        
        # Check output shape
        expected_shape = (self.batch_size, self.tgt_seq_len, self.tgt_vocab_size)
        assert output.shape == expected_shape
        
    def test_factory_function(self):
        """Test factory function for creating transformer"""
        model = create_transformer_model(
            self.src_vocab_size, self.tgt_vocab_size,
            d_model=256, num_layers=4
        )
        
        # Check that model was created with correct parameters
        assert isinstance(model, Transformer)
        assert model.d_model == 256
        
        # Test forward pass
        output = model(self.src_tokens, self.tgt_tokens)
        expected_shape = (self.batch_size, self.tgt_seq_len, self.tgt_vocab_size)
        assert output.shape == expected_shape


class TestTransformerGradients:
    """Test gradient flow through transformer"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.batch_size = 2
        self.src_seq_len = 6
        self.tgt_seq_len = 4
        self.src_vocab_size = 100
        self.tgt_vocab_size = 100
        
        # Create small model for gradient testing
        self.transformer = Transformer(
            self.src_vocab_size, self.tgt_vocab_size, 
            d_model=64, num_layers=1, num_heads=4, d_ff=128
        )
        
        # Create test inputs
        self.src_tokens = torch.randint(0, self.src_vocab_size, (self.batch_size, self.src_seq_len))
        self.tgt_tokens = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.tgt_seq_len))
        
    def test_gradient_flow(self):
        """Test that gradients flow properly through the model"""
        # Forward pass
        output = self.transformer(self.src_tokens, self.tgt_tokens)
        
        # Create dummy loss
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist for all parameters
        for name, param in self.transformer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.all(param.grad == 0), f"Zero gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])