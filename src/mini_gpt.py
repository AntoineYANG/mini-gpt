import torch
from torch import nn
import torch.nn.functional as F
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

        self.vocab_size = vocab_size
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
    def estimate_loss(self, criterion, get_batch_from_split, eval_iters):
        was_training = self.training

        if was_training:
            self.eval()

        losses = {}
        
        for split in ["train", "val"]:
            split_loss = torch.zeros(eval_iters)

            for k in range(eval_iters):
                x, y = get_batch_from_split(split)
                logits = self(x)
                B, T, C = logits.shape
                loss = criterion(
                    logits.view(B*T, C),
                    y.view(-1)
                )   # loss = -log(P(y|x)) averaged over the batch and time dimension
                split_loss[k] = loss.item()

            losses[split] = split_loss.mean().item()

        if was_training:
            self.train()

        return losses

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):

        for _ in range(max_new_tokens):

            logits = self(idx[:,-self.block_size:])  # (B, T, vocab_size)

            next_logits = logits[:, -1]  # (B, vocab_size)

            probs = F.softmax(
                next_logits / temperature,
                dim=-1
            )  # (B, vocab_size)

            next_token = torch.multinomial(
                probs,
                num_samples=1
            )  # (B, 1)

            idx = torch.cat(
                [
                    idx,
                    next_token
                ],
                dim=1
            )  # (B, T+1)

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
