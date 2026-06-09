import torch
import torch.nn as nn
from torch.optim import AdamW

from tokenizer import encode, chars
from dataset import slide_window
from mini_gpt import MiniGPT


# =========================
# Hyper Parameters
# =========================

VOCAB_SIZE = len(chars)
BLOCK_SIZE = 4

N_EMBD = 8
N_HEAD = 2
N_LAYER = 2

LR = 1e-3
EPOCHS = 1000


# =========================
# Prepare Data
# =========================

text = "I love NLP"

tokens = encode(text)

samples = slide_window(
    tokens,
    BLOCK_SIZE
)

# TODO:
# x_train
# y_train


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

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for x, y in samples:

        # ------------------
        # tensor化
        # ------------------

        x = torch.tensor(
            x,
            dtype=torch.long,
            device=device
        )

        y = torch.tensor(
            y,
            dtype=torch.long,
            device=device
        )

        # ------------------
        # Forward
        # ------------------

        logits = model(x)

        # logits:
        # (T, vocab_size)

        # y:
        # (T,)

        loss = criterion(
            logits.view(-1, VOCAB_SIZE),
            y.view(-1)
        )

        # ------------------
        # Backward
        # ------------------

        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()

        total_loss += loss.item()

    # ------------------
    # Logging
    # ------------------

    if epoch % 100 == 0:
        print(
            f"epoch={epoch} "
            f"loss={total_loss:.4f}"
        )

        pred = logits.argmax(dim=-1)

        print("pred =", pred.tolist())
        print("gold =", y.tolist())


# =========================
# Save Model
# =========================

# TODO:
torch.save(
    model.state_dict(),
    "mini_gpt.pth"
)
