import torch

from mini_gpt import MiniGPT
from tokenizer import encode, decode

model = MiniGPT()

model.load_state_dict(
    torch.load("mini_gpt.pth")
)

model.eval()

g = model.generate(
    idx=torch.tensor(encode("I")),
    max_new_tokens=20
)
print(g)
print(decode(g.tolist()))
