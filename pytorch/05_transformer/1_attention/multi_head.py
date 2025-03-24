import torch
import math
import torch.nn as nn
import numpy as np

embed_dim = 768
head_cnt = 12
head_dim = embed_dim // head_cnt
assert head_dim * head_cnt == embed_dim

seq_len = 5

tokens = torch.randn(1, seq_len, embed_dim)     # batch, tokens, embedding
Wq = torch.randn(embed_dim, head_cnt * head_dim) / math.sqrt(embed_dim)
Wk = torch.randn(embed_dim, head_cnt * head_dim) / math.sqrt(embed_dim)
Wv = torch.randn(embed_dim, head_cnt * head_dim) / math.sqrt(embed_dim)

batch, token_num, _ = tokens.shape
qis = torch.einsum("BSE,EH->BSH", tokens, Wq)
kis = torch.einsum("BTE,EH->BTH", tokens, Wk)
vis = torch.einsum("BTE,EH->BTH", tokens, Wv)
# split the single hidden dim into the heads
qis_mh = qis.view(batch, token_num, head_cnt, head_dim)
kis_mh = kis.view(batch, token_num, head_cnt, head_dim)
vis_mh = vis.view(batch, token_num, head_cnt, head_dim)

score_mat_mh = torch.einsum("BSHD,BTHD->BHST",qis_mh,kis_mh)
att_mat_mh = nn.functional.softmax(score_mat_mh / math.sqrt(head_dim), dim=-1)
zis_mh = torch.einsum("BCST,BTCH->BSCH", att_mat_mh, vis_mh)
zis = zis_mh.reshape(batch, token_num, head_cnt * head_dim)

# raw attention score of the 1st attention head
assert (torch.allclose(score_mat_mh[0, 1], qis_mh[0,:,1] @ kis_mh[0,:,1,:].T)) 

assert tokens.shape == (1, 5, 768)
print(qis_mh.shape)
print(kis_mh.shape)
print(vis_mh.shape)
print(att_mat_mh.shape)
print(zis_mh.shape)
print(zis.shape)

mha = nn.MultiheadAttention(embed_dim, head_cnt, batch_first=True,)
print(mha.in_proj_bias.shape)
mha.in_proj_weight.data = torch.cat([Wq, Wk, Wv], dim=1).T

attn_out, attn_weights = mha(tokens, tokens, tokens, average_attn_weights=False,)
assert torch.allclose(att_mat_mh, attn_weights, atol=1e-6, rtol=1e-6)

print(mha.out_proj)
assert torch.allclose(attn_out, mha.out_proj(zis), atol=1e-6, rtol=1e-6)

