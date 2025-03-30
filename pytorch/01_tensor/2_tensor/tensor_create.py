import torch
import numpy as np

data = [[1, 2], [3, 4]]
tensor_from_list = torch.tensor(data)
print('Tensor from list:', tensor_from_list)

numpy_array = np.array(data)
tensor_from_array = torch.tensor(numpy_array)
print('Tensor from array:', tensor_from_array)

tensor_zeros = torch.zeros(3, 3)
print(tensor_zeros)
tensor_ones = torch.ones(2, 2)
print(tensor_ones)
tensor_empty = torch.empty(2, 2)
print(tensor_empty)

tensor_rand = torch.rand(2, 3)
print(tensor_rand)
tensor_randn = torch.randn(2, 3)
print(tensor_randn)

tensor_arange = torch.arange(0, 10)
assert (tensor_arange.numpy() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).all()
tensor_linspace = torch.linspace(0, 1, steps=5)
assert torch.allclose(tensor_linspace,
                      torch.Tensor([0.0000, 0.2500, 0.5000, 0.7500, 1.0000]))

tensor_int = torch.tensor([1, 2, 3], dtype=torch.int64)
tensor_gpu = torch.tensor([1, 2, 3], device='cuda')
print(tensor_gpu)
