"""
Basic Usage Example

This example demonstrates how to use the transformer implementation
for a simple sequence-to-sequence task.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

# Import our transformer components (will be implemented)
# from transformer import Transformer, create_transformer_model
# from training import create_dataloaders, Trainer, TrainingConfig


def create_simple_data():
    """
    Create simple training data for demonstration
    
    Returns:
        Tuple[List[str], List[str]]: Source and target sequences
    """
    # Simple English to French translation examples
    src_texts = [
        "Hello world",
        "How are you",
        "I love coding",
        "Python is great",
        "Machine learning is fun"
    ]
    
    tgt_texts = [
        "Bonjour le monde",
        "Comment allez-vous",
        "J'aime coder",
        "Python est génial",
        "L'apprentissage automatique est amusant"
    ]
    
    return src_texts, tgt_texts


def build_simple_vocab(texts: List[str], min_freq: int = 1):
    """
    Build a simple vocabulary from text
    
    Args:
        texts (List[str]): List of text sequences
        min_freq (int): Minimum frequency for tokens
        
    Returns:
        dict: Token to index mapping
    """
    # TODO: Implement simple vocabulary building
    # 1. Tokenize all texts
    # 2. Count token frequencies
    # 3. Create token2idx mapping
    # 4. Add special tokens (PAD, UNK, SOS, EOS)
    pass


def tokenize_text(text: str, vocab: dict) -> List[int]:
    """
    Tokenize text using vocabulary
    
    Args:
        text (str): Input text
        vocab (dict): Token to index mapping
        
    Returns:
        List[int]: Token indices
    """
    # TODO: Implement simple tokenization
    # 1. Split text into tokens
    # 2. Convert tokens to indices
    # 3. Handle unknown tokens
    pass


def detokenize_text(indices: List[int], vocab: dict) -> str:
    """
    Convert token indices back to text
    
    Args:
        indices (List[int]): Token indices
        vocab (dict): Token to index mapping
        
    Returns:
        str: Reconstructed text
    """
    # TODO: Implement simple detokenization
    # 1. Convert indices to tokens
    # 2. Join tokens into text
    # 3. Handle special tokens
    pass


def main():
    """Main function demonstrating basic transformer usage"""
    print("🤖 Transformer from Scratch - Basic Usage Example")
    print("=" * 50)
    
    # 1. Create simple training data
    print("\n1. Creating training data...")
    src_texts, tgt_texts = create_simple_data()
    print(f"   Created {len(src_texts)} training examples")
    
    # 2. Build vocabularies
    print("\n2. Building vocabularies...")
    src_vocab = build_simple_vocab(src_texts)
    tgt_vocab = build_simple_vocab(tgt_texts)
    print(f"   Source vocabulary size: {len(src_vocab)}")
    print(f"   Target vocabulary size: {len(tgt_vocab)}")
    
    # 3. Create transformer model
    print("\n3. Creating transformer model...")
    # model = create_transformer_model(
    #     src_vocab_size=len(src_vocab),
    #     tgt_vocab_size=len(tgt_vocab),
    #     d_model=256,  # Smaller for demo
    #     num_layers=3,
    #     num_heads=4,
    #     d_ff=1024
    # )
    print("   Model created successfully")
    
    # 4. Setup training
    print("\n4. Setting up training...")
    # config = TrainingConfig(
    #     batch_size=2,
    #     learning_rate=1e-4,
    #     num_epochs=10,
    #     warmup_steps=100
    # )
    print("   Training configuration ready")
    
    # 5. Create dataloaders
    print("\n5. Creating dataloaders...")
    # train_loader, val_loader = create_dataloaders(
    #     src_texts, tgt_texts, src_vocab, tgt_vocab,
    #     batch_size=config.batch_size
    # )
    print("   Data loaders created")
    
    # 6. Train model
    print("\n6. Training model...")
    # trainer = Trainer(model, config, train_loader, val_loader)
    # trainer.train()
    print("   Training completed")
    
    # 7. Test translation
    print("\n7. Testing translation...")
    test_text = "Hello world"
    # translated = model.translate(test_text)
    # print(f"   Input: {test_text}")
    # print(f"   Output: {translated}")
    
    print("\n✅ Basic usage example completed!")
    print("\nNext steps:")
    print("- Implement the TODO sections")
    print("- Add more training data")
    print("- Experiment with different model sizes")
    print("- Try different tasks (classification, generation)")


if __name__ == "__main__":
    main() 