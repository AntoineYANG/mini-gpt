import torch
import torch.nn as nn
from torch.optim import AdamW

from tokenizer import encode, chars
from dataset import build_dataset, get_batch
from mini_gpt import MiniGPT


# =========================
# Hyper Parameters
# =========================

VOCAB_SIZE = len(chars)
BLOCK_SIZE = 4
BATCH_SIZE = 2

N_EMBD = 8
N_HEAD = 2
N_LAYER = 2

LR = 1e-3
MAX_STEPS = 10000


# =========================
# Prepare Data
# =========================

text = "I love NLP"

tokens = encode(text)

xs, ys = build_dataset(tokens, BLOCK_SIZE)


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
device = torch.device("cpu")
print(f"device: {device}")


# =========================
# Model
# =========================

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER
).to(device)


# =========================
# Loss
# =========================

criterion = nn.CrossEntropyLoss()


# =========================
# Optimizer
# =========================

optimizer = AdamW(
    model.parameters(),
    lr=LR
)


# =========================
# Training Loop
# =========================

for step in range(MAX_STEPS):

    model.train()

    xb, yb = get_batch(xs, ys, BATCH_SIZE)

    # ------------------
    # Forward
    # ------------------

    logits = model(xb)

    # logits:
    # (B, T, vocab_size)

    # yb:
    # (B, T)

    loss = criterion(
        logits.view(-1, VOCAB_SIZE),    # (B*T, vocab_size)
        yb.view(-1)                # (B*T,)
    )

    # ------------------
    # Backward
    # ------------------

    optimizer.zero_grad()
        
    loss.backward()
        
    optimizer.step()

    total_loss = loss.item()

    # ------------------
    # Logging
    # ------------------

    if step % 100 == 0:
        print(
            f"step={step} "
            f"loss={total_loss:.4f}"
        )


# =========================
# Save Model
# =========================

torch.save(
    model.state_dict(),
    "mini_gpt.pth"
)
