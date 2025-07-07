#!/usr/bin/env python3
"""
Simple test script to verify transformer implementation works
"""

import torch
import sys
import os

# Add current directory to path to import our modules
sys.path.insert(0, os.path.dirname(__file__))

from transformer.model import Transformer, create_transformer_model
from transformer.encoder import TransformerEncoder
from transformer.decoder import TransformerDecoder, generate_square_subsequent_mask


def test_encoder_decoder():
    """Test basic encoder-decoder functionality"""
    print("🧪 Testing Encoder-Decoder Architecture...")
    
    # Test parameters
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 2
    
    try:
        # Create transformer model
        print("  Creating transformer model...")
        transformer = Transformer(
            src_vocab_size, tgt_vocab_size, d_model, 
            num_layers, num_heads, d_ff, dropout=0.1
        )
        print("  ✅ Model created successfully")
        
        # Create test inputs (token indices)
        src_tokens = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
        tgt_tokens = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
        
        print("  Testing forward pass...")
        # Test forward pass
        output = transformer(src_tokens, tgt_tokens)
        expected_shape = (batch_size, tgt_seq_len, tgt_vocab_size)
        
        assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"
        print(f"  ✅ Forward pass successful. Output shape: {output.shape}")
        
        # Test encoding
        print("  Testing encoding...")
        encoder_output, enc_attn = transformer.encode(src_tokens)
        expected_enc_shape = (batch_size, src_seq_len, d_model)
        
        assert encoder_output.shape == expected_enc_shape, f"Expected encoder shape {expected_enc_shape}, got {encoder_output.shape}"
        assert len(enc_attn) == num_layers, f"Expected {num_layers} attention layers, got {len(enc_attn)}"
        print(f"  ✅ Encoding successful. Encoder output shape: {encoder_output.shape}")
        
        # Test decoding
        print("  Testing decoding...")
        decoder_output, dec_attn = transformer.decode(tgt_tokens, encoder_output)
        expected_dec_shape = (batch_size, tgt_seq_len, d_model)
        
        assert decoder_output.shape == expected_dec_shape, f"Expected decoder shape {expected_dec_shape}, got {decoder_output.shape}"
        assert len(dec_attn[0]) == num_layers, f"Expected {num_layers} self-attention layers, got {len(dec_attn[0])}"
        assert len(dec_attn[1]) == num_layers, f"Expected {num_layers} encoder-decoder attention layers, got {len(dec_attn[1])}"
        print(f"  ✅ Decoding successful. Decoder output shape: {decoder_output.shape}")
        
        # Test generation
        print("  Testing generation...")
        generated = transformer.generate(src_tokens, max_len=5, start_token=1, end_token=2)
        
        assert generated.shape[0] == batch_size, f"Expected batch size {batch_size}, got {generated.shape[0]}"
        assert generated.shape[1] <= 6, f"Expected max length 6, got {generated.shape[1]}"
        assert torch.all(generated[:, 0] == 1), "All sequences should start with start_token"
        print(f"  ✅ Generation successful. Generated shape: {generated.shape}")
        
        # Test with masks
        print("  Testing with masks...")
        src_mask = torch.zeros(batch_size, src_seq_len, src_seq_len)
        src_mask[:, :, -2:] = float('-inf')  # Mask last 2 positions
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
        
        masked_output = transformer(src_tokens, tgt_tokens, src_mask, tgt_mask)
        assert masked_output.shape == expected_shape, f"Expected shape {expected_shape}, got {masked_output.shape}"
        print(f"  ✅ Masking successful. Masked output shape: {masked_output.shape}")
        
        # Test gradient flow
        print("  Testing gradient flow...")
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        grad_count = 0
        for name, param in transformer.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_count += 1
        
        assert grad_count > 0, "No gradients found"
        print(f"  ✅ Gradient flow successful. {grad_count} parameters have gradients")
        
        print("🎉 All tests passed! Your transformer implementation is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_components():
    """Test individual components"""
    print("\n🔧 Testing Individual Components...")
    
    try:
        # Test causal mask
        print("  Testing causal mask generation...")
        size = 5
        mask = generate_square_subsequent_mask(size)
        assert mask.shape == (size, size), f"Expected shape ({size}, {size}), got {mask.shape}"
        
        # Verify upper triangular structure
        for i in range(size):
            for j in range(size):
                if j > i:
                    assert mask[i, j] == float('-inf'), f"Position ({i}, {j}) should be -inf"
                else:
                    assert mask[i, j] == 0.0, f"Position ({i}, {j}) should be 0.0"
        
        print("  ✅ Causal mask generation successful")
        
        # Test factory function
        print("  Testing factory function...")
        model = create_transformer_model(100, 100, d_model=64, num_layers=1)
        assert isinstance(model, Transformer), "Factory function should return Transformer instance"
        assert model.d_model == 64, "Model should have correct d_model"
        print("  ✅ Factory function successful")
        
        print("🎉 All component tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 Testing Transformer Implementation")
    print("=" * 50)
    
    # Test components first
    component_success = test_components()
    
    # Test full implementation
    full_success = test_encoder_decoder()
    
    print("\n" + "=" * 50)
    if component_success and full_success:
        print("🎉 ALL TESTS PASSED! Your transformer is ready to use.")
        print("\nNext steps:")
        print("1. Train the model on your dataset")
        print("2. Implement attention mechanisms if not done already")
        print("3. Add any task-specific heads (classification, generation, etc.)")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())