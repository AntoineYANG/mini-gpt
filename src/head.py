import torch
from torch import nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.n_embd = n_embd
        self.head_size = head_size

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

    def forward(self, x):
        K = self.key(x) # (B,T,C)
        Q = self.query(x) # (B,T,C)
        V = self.value(x) # (B,T,C)
        scores = Q @ K.transpose(-2, -1) / self.head_size ** 0.5
        _,T,__ = x.shape
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.2):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.heads = nn.ModuleList([
            Head(n_embd, n_embd // n_head)
            for _ in range(n_head)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        out = self.dropout(out)
        return out

if __name__ == "__main__":
    x = torch.randn(2, 4, 8)

    head = Head(n_embd=8, head_size=4)
    out = head(x)
    print(out.shape)

    mha = MultiHeadAttention(
        n_embd=8,
        n_head=2,
        dropout=0.0
    )

    out = mha(x)
    print(out.shape)

    pass
