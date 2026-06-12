import torch
import torch.nn as nn
from torch.optim import AdamW

from tokenizer import Tokenizer
from dataset import get_batch
from mini_gpt import MiniGPT


# =========================
# Hyper Parameters
# =========================

VOCAB_SIZE = None
BLOCK_SIZE = 128
BATCH_SIZE = 32

N_EMBD = 128
N_HEAD = 8
N_LAYER = 8

LR = 3e-4
MAX_STEPS = 10000
EVAL_INTERVAL = 100


# =========================
# Device
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"device: {device}")


# =========================
# Prepare Data
# =========================

with open("data/input.txt", "r") as f:
    text = f.read()

tokenizer = Tokenizer(text)

VOCAB_SIZE = tokenizer.vocab_size()

tokens = torch.tensor(
    tokenizer.encode(text),
    dtype=torch.long
)

n = int(0.9 * len(tokens))

train_data = tokens[:n]
val_data = tokens[n:]

def get_batch_from_split(split):
    if split == "train":
        return get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
    elif split == "val":
        return get_batch(val_data, BLOCK_SIZE, BATCH_SIZE, device)
    return None, None


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

best_val_loss = float('inf')

train_history = []
val_history = []

for step in range(MAX_STEPS):

    model.train()

    x, y = get_batch_from_split("train")

    # ------------------
    # Forward
    # ------------------

    logits, loss = model(x, y)

    # logits:
    # (B, T, vocab_size)

    # y:
    # (B, T)

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

    if step % EVAL_INTERVAL == 0:
        losses = model.estimate_loss(
            criterion,
            get_batch_from_split,
            eval_iters=100
        )
        train_history.append(losses['train'])
        val_history.append(losses['val'])
        print(
            f"step {step}: "
            f"train {losses['train']:.4f}, val {losses['val']:.4f}"
        )
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            torch.save(
                model.state_dict(),
                "mini_gpt--tiny_shakespeare.best_model.pth"
            )


# =========================
# Save Model
# =========================

torch.save(
    model.state_dict(),
    "mini_gpt--tiny_shakespeare.pth"
)

### save training history plot
import matplotlib.pyplot as plt
plt.plot(train_history, label="train")
plt.plot(val_history, label="val")
plt.xlabel("Evaluation Step")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig("training_history.png")
