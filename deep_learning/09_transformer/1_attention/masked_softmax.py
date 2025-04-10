import torch

def sequence_mask(x, valid_len, value=0):
    # Perform softmax operation by masking elements on the last axis.
    maxlen = x.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32)[None, :] < valid_len[:, None]
    x[~mask] = value
    return x

def masked_softmax(x, valid_lens):
    # Perform softmax operation by masking elements on the last axis.
    # x: 3D tensor, valid_lens: 1D or 2D tensor
    if valid_lens is None:
        return torch.nn.functional.softmax(x, dim=-1)
    else:
        shape = x.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        x = sequence_mask(x.reshape(-1, shape[-1]), valid_lens, value=-1e6)
        return torch.nn.functional.softmax(x.reshape(shape), dim=-1)

if __name__ == '__main__':
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))
