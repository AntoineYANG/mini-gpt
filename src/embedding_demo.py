import torch
import torch.nn as nn

vocab_size = 9#1000
embedding_dim = 8

embedding = nn.Embedding(vocab_size, embedding_dim)

tokens = torch.tensor([1,2,3])

out = embedding(tokens)

print(out.shape)
print(out)