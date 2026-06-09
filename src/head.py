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
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        scores = Q @ K.transpose(-2, -1) / self.head_size ** 0.5
        T = x.shape[0]
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.heads = nn.ModuleList([
            Head(n_embd, n_embd // n_head)
            for _ in range(n_head)
        ])

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads],
            dim=-1
        )
        return out

if __name__ == "__main__":
    x = torch.randn(4, 8)

    mha = MultiHeadAttention(
        n_embd=8,
        n_head=2
    )

    out = mha(x)
    print(out.shape)

    pass
