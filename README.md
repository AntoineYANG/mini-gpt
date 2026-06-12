Implemented from scratch:

✓ Character Tokenizer
✓ Sliding Window Dataset
✓ Token Embedding
✓ Positional Embedding
✓ Causal Self Attention
✓ Multi Head Attention
✓ Feed Forward Network
✓ Transformer Block
✓ GPT Model
✓ Cross Entropy Training

## Features

Tokenizer

Embedding

Position Embedding

Self Attention

Multi Head Attention

LayerNorm

Residual

FeedForward

Transformer Block

GPT

Batch Training

Validation

Sampling Generation


## Data Preparation

Tiny Shakespeare

```
curl -o data/tinyshakespeare.txt \
https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```


## Experimental Results

### Tiny Shakespeare

```python
BLOCK_SIZE = 128
BATCH_SIZE = 32
LR = 3e-4
MAX_STEPS = 10000
```

|n_embd|n_head|n_layer|params|val loss|
|---|---|---|---|---|
|64|4|4|199,233|1.7614|
|128|4|4|758,849|1.6105|
|**256**|4|4|**2,959,425**|**1.5127**|
|512|4|4|11,685,953|1.5841|

|n_embd|n_head|n_layer|params|val loss|
|---|---|---|---|---|
|256|2|4|2,959,425|1.5254|
|256|4|4|2,959,425|1.5127|
|256|6|4|||
|256|8|4|||
