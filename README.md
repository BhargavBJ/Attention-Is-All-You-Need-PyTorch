# Attention Is All You Need – PyTorch

This repository contains a **from-scratch PyTorch implementation** of the Transformer architecture  
introduced in the paper **"Attention Is All You Need" (Vaswani et al., NeurIPS 2017)**.

Implemented fully inside a **single Jupyter/Colab notebook**, without using `torch.nn.Transformer`.  
Every component (Multi-Head Attention, Encoder, Decoder, Positional Encoding, Masks) is built step by step.

---

## 📘 Notebook
- **`Attention_Is_All_You_Need.ipynb`** → The complete project notebook.  
  - Transformer Encoder–Decoder model  
  - Scaled Dot-Product Attention & Multi-Head Attention  
  - Positional Encoding (sinusoidal)  
  - Source & Target masks  
  - Verified forward pass on toy data  

You can try it directly in Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HoRN3YZcsVqcRYzO_LevdZ1o9oi_Vch9?usp=sharing)


---

## 🚀 Example Usage
```python
import torch
from transformer import Transformer  # or copy class from notebook

device = "cuda" if torch.cuda.is_available() else "cpu"

src = torch.tensor([[1,5,6,4,3,9,5,2,0],
                    [1,8,7,3,4,5,6,7,2]]).to(device)

trg = torch.tensor([[1,7,4,3,5,9,2,0],
                    [1,5,6,2,4,7,6,2]]).to(device)

model = Transformer(
    src_vocab_size=10,
    trg_vocab_size=10,
    src_pad_idx=0,
    trg_pad_idx=0,
    device=device
).to(device)
out = model(src, trg[:, :-1])
print(out.shape)  # (batch_size, trg_len, vocab_size)
```
