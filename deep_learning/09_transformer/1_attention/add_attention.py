import torch
from torch import nn
from masked_softmax import masked_softmax
from show_heatmap import show_heatmap

class AdditiveAttention(nn.Module):
    '''Additive attention.'''
    def __init__(self, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = nn.LazyLinear(num_hiddens, bias=False)
        self.W_q = nn.LazyLinear(num_hiddens, bias=False)
        self.w_v = nn.LazyLinear(1, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        assert queries.shape == (2, 1, 8)
        assert keys.shape == (2, 10, 8)
        # After dimension expansion, shape of queries: (batch_size, no. of
        # queries, 1, num_hiddens) and shape of keys: (batch_size, 1, no. of
        # key-value pairs, num_hiddens). Sum them up with broadcasting
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        assert features.shape == (2, 1, 10, 8)
        features = torch.tanh(features)
        # There is only one output of self.w_v, so we remove the last
        # one-dimensional entry from the shape. Shape of scores: (batch_size,
        # no. of queries, no. of key-value pairs)
        scores = self.w_v(features).squeeze(-1)
        assert scores.shape == (2, 1, 10)
        assert values.shape == (2, 10, 4)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # Shape of values: (batch_size, no. of key-value pairs, value
        # dimension)
        return torch.bmm(self.dropout(self.attention_weights), values)

if __name__ == '__main__':
    # 2 个样本，每个样本 1 个查询向量，维度是 20
    queries = torch.normal(0, 1, (2, 1, 20))
    keys = torch.normal(0, 1, (2, 10, 2))
    values = torch.normal(0, 1, (2, 10, 4))
    valid_lens = torch.tensor([2, 6])
    attention = AdditiveAttention(num_hiddens=8, dropout=0.1)
    attention.eval()
    assert attention(queries, keys, values, valid_lens).shape == (2, 1, 4)

    show_heatmap(attention.attention_weights.detach().reshape((1, 1, 2, 10)),
                 x_label='Keys', y_label='Queries')
