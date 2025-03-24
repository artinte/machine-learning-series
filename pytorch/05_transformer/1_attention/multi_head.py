import torch

embed_dim = 768
head_cnt = 12
head_dim = embed_dim // head_cnt
assert head_dim * head_cnt == embed_dim

seq_len = 5

tokens = torch.randn(1, 5, embed_dim)
