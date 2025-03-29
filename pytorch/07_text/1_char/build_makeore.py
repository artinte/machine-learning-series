import torch
import matplotlib.pyplot as plt

words = open('names.txt', 'r').read().splitlines()
print(words[:10])
print(len(words))
print(min(len(w) for w in words))
print(max(len(w) for w in words))

for w in words[:1]:
    for ch1, ch2 in zip(w, w[1:]):
        print(ch1, ch2)

for w in words[:1]:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        print(ch1, ch2)

b = {}
for w in words[:1]:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        bigram = (ch1, ch2)
        b[bigram] = b.get(bigram, 0) + 1        
print(b)

sorted(b.items(), key=lambda kv: kv[1])


N = torch.zeros((28, 28), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i for i,s in enumerate(chars)}
stoi['<S>'] = 26
stoi['<E>'] = 27

for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

print(N)
  
plt.imshow(N)
plt.show()

itos = {i:s for i,s in enumerate(chars)}
itos[26] = '<S>'
itos[27] = '<E>'
plt.figure(figsize=(16, 8))
ax = plt.gca()
ax.imshow(N, cmap='Blues', aspect='auto')
for i in range(28):
    for j in range(28):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)
plt.show()

N = torch.zeros((27, 27), dtype=torch.int32)
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i, s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s, i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1

plt.figure(figsize=(16, 8))
ax = plt.gca()
ax.imshow(N, cmap='Blues', aspect='auto')
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i, j].item(), ha='center', va='top', color='gray')
plt.axis('off')
plt.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.04)
plt.show()

print(N[0, :])
p = N[0].float()
p = p / p.sum()
print(p.sum())

g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
print(itos[ix])

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3, generator=g)
p = p / p.sum()
print(p)

result = torch.multinomial(p, num_samples=20, replacement=True, generator=g)
print(result)

g = torch.Generator().manual_seed(2147483647)
N = N.float()
for i in range(5):
    out = []
    ix = 0
    while True:
        p = N[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))
