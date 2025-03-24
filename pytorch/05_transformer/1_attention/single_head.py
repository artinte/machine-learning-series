import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

embed_dim = 256     # 词嵌入大小
head_dim = 64       # 单头注意力大小
seq_len = 5

tokens = torch.rand(1, seq_len, embed_dim)        # batch, tokens, embedding
Wq = torch.rand(embed_dim, head_dim) / math.sqrt(embed_dim)
Wk = torch.rand(embed_dim, head_dim) / math.sqrt(embed_dim)
Wv = torch.rand(embed_dim, embed_dim) / math.sqrt(embed_dim)

qis = torch.einsum('BSE,EH->BSH', tokens, Wq)   # batch x seq_len x head_dim
kis = torch.einsum('BTE,EH->BTH', tokens, Wk)   # batch x seq_len x head_dim
vis = torch.einsum('BTE,EF->BTF', tokens, Wv)   # batch x seq_len x embed_dim

score_mat = torch.einsum('BSH,BTH->BST', qis, kis) # output: batch x seq_len (Query) x seq_len (Key)

assert(torch.isclose(score_mat[0,1,2], qis[0,1,:]@kis[0,2,:]))
assert(torch.isclose(score_mat[0,3,4], qis[0,3,:]@kis[0,4,:]))
assert(torch.isclose(score_mat[0,2,2], qis[0,2,:]@kis[0,2,:]))

att_mat = nn.functional.softmax(score_mat / math.sqrt(head_dim), dim=2)
zis = torch.einsum('BST,BTF->BSF', att_mat, vis)

attn_torch = nn.functional.scaled_dot_product_attention(qis, kis, vis)
assert(torch.allclose(attn_torch, zis, atol=1e-6, rtol=1e-6))
