from torch import nn
from head import MultiHeadAttention
from feed_forward import FeedForward

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head)
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # Pre-LN Transformer
        x = x + self.ffwd(self.ln2(x))
        return x

if __name__ == "__main__":
    block = TransformerBlock(
        n_embd=8,
        n_head=2
    )
    import torch
    x = torch.randn(4, 8)

    out = block(x)
    print(out.shape)
    pass