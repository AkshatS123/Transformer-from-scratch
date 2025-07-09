# 🤖 I Built a Transformer from Scratch(I'm cracked :))))

A complete PyTorch implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).

## ✅ Features

- **Multi-head attention** with proper masking
- **Encoder-decoder architecture** with residual connections
- **Positional encoding** for sequence modeling
- **Comprehensive test suite** (27 passing tests)
- **Clean, documented code** for learning

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python run_tests.py

# Try basic usage
python test_implementation.py
```

## 📂 Structure

```
transformer/
├── attention.py    # Multi-head attention mechanisms
├── encoder.py      # Encoder blocks and layers  
├── decoder.py      # Decoder blocks with causal masking
└── model.py        # Complete transformer model
```

## 🔧 Usage

```python
from transformer.model import Transformer

# Create model
model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    d_model=512,
    num_layers=6,
    num_heads=8,
    d_ff=2048
)

# Forward pass
src_tokens = torch.randint(0, 1000, (2, 10))  # (batch, seq_len)
tgt_tokens = torch.randint(0, 1000, (2, 8))
output = model(src_tokens, tgt_tokens)  # (batch, tgt_len, vocab_size)
```

## 🧪 Testing

![image](https://github.com/user-attachments/assets/d7b33ce6-6e11-461e-93de-279251448acc)


All 27 tests pass, covering:
- ✅ Attention mechanisms and masking
- ✅ Encoder-decoder architecture
- ✅ Positional encoding
- ✅ Gradient flow
- ✅ Text generation

## 📖 Key Concepts

**Attention Formula:** `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

- **Self-attention**: Q, K, V from same sequence
- **Cross-attention**: Q from target, K,V from source  
- **Causal masking**: Prevents looking at future tokens
- **Multi-head**: Parallel attention in different subspaces 
