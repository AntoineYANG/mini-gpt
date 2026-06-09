def slide_window(tokens, block_size):
    samples = []
    for i in range(len(tokens) - block_size):
        x = tokens[i:i+block_size]
        y = tokens[i+1:i+block_size+1]
        samples.append((x, y))
    return samples
