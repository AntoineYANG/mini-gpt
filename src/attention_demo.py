import torch
import torch.nn.functional as F

embedding_dim = 8

x = torch.randn(4, embedding_dim)  # (batch_size, embedding_dim)

Wq = torch.randn(embedding_dim, embedding_dim)
Wk = torch.randn(embedding_dim, embedding_dim)
Wv = torch.randn(embedding_dim, embedding_dim)

Q = x @ Wq
K = x @ Wk
V = x @ Wv

print(Q.shape)
print(K.shape)
print(V.shape)

scores = Q @ K.T / embedding_dim**0.5
"""
Why divide by √dk in attention?
  - Because the variance of the dot product grows with the dimension.
    Scaling by √dk keeps the attention logits in a reasonable range and prevents
    softmax saturation, which stabilizes gradients during training.
"""

print(scores.shape)

attn = F.softmax(scores, dim=-1)
print(attn)
print(attn.sum(dim=-1))

out = attn @ V
print(out.shape)
