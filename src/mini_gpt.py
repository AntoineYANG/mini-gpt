import torch
from torch import nn
from transformer_block import TransformerBlock

class MiniGPT(nn.Module):
    """
    Tokenizer
    ↓
    Embedding
    ↓
    Position Embedding
    ↓
    Transformer Blocks
    ↓
    LayerNorm
    ↓
    LM Head
    ↓
    Logits
    """
    def __init__(
        self,
        vocab_size=9,
        block_size=4,
        n_embd=8,
        n_head=2,
        n_layer=2
    ):
        super().__init__()

        self.block_size = block_size
        
        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )
        self.position_embedding_table = nn.Embedding(
            block_size,
            n_embd
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd,
                    n_head
                )
                for _ in range(n_layer)
            ]
        )

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx):
        _, T = idx.shape    # (B, T)

        token_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos = torch.arange(
            T,
            device=idx.device
        )  # (T,)
        pos_emb = self.position_embedding_table(pos)  # (T, n_embd) broadcast -> (1, T, n_embd)

        x = token_emb + pos_emb  # (B, T, n_embd)

        x = self.blocks(x)  # (B, T, n_embd)

        x = self.ln_f(x)  # (B, T, n_embd)

        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        for _ in range(max_new_tokens):

            logits = self(idx[-self.block_size:])  # (T, vocab_size)

            next_logits = logits[-1]  # (vocab_size,)

            next_token = torch.argmax(
                next_logits,
                dim=-1
            )  # ()

            idx = torch.cat(
                [
                    idx,
                    next_token.unsqueeze(0)
                ],
                dim=0
            )  # (T+1,)

        return idx



if __name__ == "__main__":
    model = MiniGPT(
        vocab_size=9,
        block_size=4,
        n_embd=8,
        n_head=2,
        n_layer=2
    )

    idx = torch.tensor([
        [1,2,3,4],
        [2,3,4,5]
    ])

    logits = model(idx)

    print(logits.shape)
