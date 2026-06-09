import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd

        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x):
        K = self.key(x)
        Q = self.query(x)
        V = self.value(x)
        scores = Q @ K.transpose(-2, -1) / self.n_embd ** 0.5
        T = x.shape[0]
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V
        return out


if __name__ == "__main__":
    x = torch.randn(4, 8)

    attn = SelfAttention(n_embd=8)

    out = attn(x)

    print(out.shape)
    