import torch

"""
Dataset and DataLoader
Arguments:
- tokens: (N,)
- block_size: int
- batch_size: int
- device: torch.device
Return:
- x: (B, T)
- y: (B, T)
"""
def get_batch(tokens, block_size, batch_size, device):
    ix = torch.randint(
        len(tokens) - block_size,
        (batch_size,)
    )
    x = torch.stack([
        tokens[i:i+block_size]
        for i in ix
    ]).to(device)
    y = torch.stack([
        tokens[i+1:i+block_size+1]
        for i in ix
    ]).to(device)
    return x, y

if __name__ == '__main__':
    tokens = torch.tensor([i for i in range(10)], dtype=torch.long)
    batch_size = 2
    device = torch.device("cpu")
    x, y = get_batch(tokens, 4, batch_size, device)
    print(x)
    print(y)
