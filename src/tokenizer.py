class Tokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {
            char: i
            for i, char in enumerate(self.chars)
        }
        self.itos = {
            i: char
            for i, char in enumerate(self.chars)
        }

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return "".join([self.itos[c] for c in l])

    def vocab_size(self):
        return len(self.chars)
        