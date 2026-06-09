text = "I love NLP"
chars = sorted(list(set(text)))

stoi = {
    char: i
    for i, char in enumerate(chars)
}

itos = {
    i: char
    for i, char in enumerate(chars)
}

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return "".join([itos[c] for c in l])

#encoded = encode(text)
#decoded = decode(encoded)

#print(encoded)
#print(decoded)