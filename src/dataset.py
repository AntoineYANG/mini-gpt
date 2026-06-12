import torch

def slide_window(tokens, block_size):
    samples = []
    for i in range(len(tokens) - block_size):
        x = tokens[i:i+block_size]
        y = tokens[i+1:i+block_size+1]
        samples.append((x, y))
    return samples

def build_dataset(tokens, block_size):
    xs = []
    ys = []
    for i in range(len(tokens) - block_size):
        xs.append(tokens[i:i+block_size])
        ys.append(tokens[i+1:i+block_size+1])
    xs = torch.tensor(xs, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys

def get_batch(xs, ys, batch_size):
    ix = torch.randint(
        len(xs),
        (batch_size,)
    )
    xb = xs[ix]
    yb = ys[ix]
    return xb, yb

if __name__ == '__main__':
    tokens = [i for i in range(10)]
    xs, ys = build_dataset(tokens, 4)
    batch_size = 2
    xb, yb = get_batch(xs, ys, batch_size)
    print(xb)
    print(yb)
