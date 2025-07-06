# 🤖 Transformer from Scratch

> Learning to build transformers from the ground up

## 🎯 Project Overview

This repository documents my journey of implementing a complete Transformer architecture from scratch. The goal is to deeply understand the internals of transformers by building every component ourselves.

## 🏗️ Project Structure

```
transformer-from-scratch/
├── transformer/              # Core transformer implementation
│   ├── attention.py         # Multi-head attention mechanisms
│   ├── encoder.py           # Encoder blocks and layers
│   ├── decoder.py           # Decoder blocks with masking
│   ├── model.py             # Complete transformer model
│   └── __init__.py          # Package initialization
├── training/                # Training infrastructure
│   ├── train.py             # Training and evaluation loops
│   ├── dataset.py           # Data loading and preprocessing
│   ├── optimizer.py         # Custom optimizers and schedulers
│   └── __init__.py          # Package initialization
├── experiments/             # Real-world applications
│   ├── translation/         # Machine translation tasks
│   ├── classification/      # Text classification
│   └── generation/          # Text generation and language modeling
├── tests/                   # Test suite
├── examples/                # Usage examples
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/transformer-from-scratch.git
cd transformer-from-scratch

# Install dependencies
pip install -r requirements.txt
```

## 📚 Learning Goals

- [x] Understand attention mechanisms
- [ ] Implement encoder-decoder architecture
- [ ] Build training pipeline
- [ ] Apply to real-world tasks

## 🔬 Key Resources

- [**"Attention Is All You Need"**](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [**"The Illustrated Transformer"**](http://jalammar.github.io/illustrated-transformer/) - Visual guide
- [**PyTorch Tutorials**](https://pytorch.org/tutorials/) - Framework documentation

## 🧠 Attention Mechanism Understanding

**Core Formula:** `Attention(Q,K,V) = softmax(QK^T/√d_k)V`

**Key Insights:**
- **Query (Q)**: What we're looking for
- **Key (K)**: What we're matching against
- **Value (V)**: What we're retrieving
- **√d_k scaling**: Prevents softmax saturation for stable gradients
- **Multi-head**: Allows attending to different representation subspaces

---

*Learning in progress...* 