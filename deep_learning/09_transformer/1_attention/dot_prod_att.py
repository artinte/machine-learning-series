import math
import torch
from torch import nn
from masked_softmax import masked_softmax
from show_heatmap import show_heatmap

class DotProductAttention(nn.Module):
    '''Scaled dot product attention.'''
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    # Shape of queries: (batch_size, no. of queries, d)
    # Shape of keys: (batch_size, no. of key-value pairs, d)
    # Shape of values: (batch_size, no. of key-value pairs, value dimension)
    # Shape of valid_lens: (batch_size,) or (batch_size, no. of queries)
    def forward(self, queries, keys, values, valid_lens=None):    
        d = queries.shape[-1]
        # Swap the last two dimensions of keys with keys.transpose(1, 2).
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

if __name__ == '__main__':
    queries = torch.normal(0, 1, (2, 1, 2))
    keys = torch.normal(0, 1, (2, 10, 2))
    values = torch.normal(0, 1, (2, 10, 4))
    valid_lens = torch.tensor([2, 6])
    
    attention = DotProductAttention(dropout=0.1)
    attention.eval()
    assert attention(queries, keys, values, valid_lens).shape == (2, 1, 4)
    
    show_heatmap(attention.attention_weights.reshape((1, 1, 2, 10)),
                 x_label='Keys', y_label='Queries')

        