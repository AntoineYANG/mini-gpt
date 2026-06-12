import torch

from mini_gpt import MiniGPT
from tokenizer import Tokenizer

VOCAB_SIZE = None
BLOCK_SIZE = 128

N_EMBD = 128
N_HEAD = 8
N_LAYER = 8

with open("data/input.txt", "r") as f:
    text = f.read()

tokenizer = Tokenizer(text)

VOCAB_SIZE = tokenizer.vocab_size()

model = MiniGPT(
    vocab_size=VOCAB_SIZE,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=N_HEAD,
    n_layer=N_LAYER
)

model.load_state_dict(
    torch.load("mini_gpt--tiny_shakespeare.best_model.pth")
)

model.eval()

context = torch.tensor(
    tokenizer.encode("First"),
    dtype=torch.long
).unsqueeze(0)  # (1, T)
out = model.generate(
    idx=context,
    max_new_tokens=300,
    temperature=0.8
)
print(out.tolist())
print(tokenizer.decode(out.tolist()[0]))
