import torch
import torch.nn as nn

vocab_size = 9
n_embd = 8
block_size = 4

token_embedding = nn.Embedding(vocab_size, n_embd)
position_embedding = nn.Embedding(block_size, n_embd)

tokens = torch.tensor([1, 2, 3, 4])

token_emb = token_embedding(tokens)

positions = torch.arange(block_size)

pos_emb = position_embedding(positions)

x = token_emb + pos_emb

print(token_emb.shape)
print(pos_emb.shape)
print(x.shape)
