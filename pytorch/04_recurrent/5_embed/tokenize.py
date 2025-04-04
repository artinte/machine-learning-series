sentence = 'The wide road shimmered in the hot sun'

tokens = list(sentence.lower().split())
assert len(tokens) == 8

vocab, index = {}, 1    # starting indexing from 1
vocab['<pad>'] = 0  # add a padding token
for token in tokens:
    if token not in vocab:
        vocab[token] = index
        index += 1
vocab_size = len(vocab)
print(vocab)

inverse_vocab = {index: token for token, index in vocab.items()}
print(inverse_vocab)

example_sequence = [vocab[word] for word in tokens]
print(example_sequence)
