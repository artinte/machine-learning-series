import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

assert ((a + b) == torch.tensor([5, 7, 9])).all()
assert ((a * b) == torch.tensor([4, 10, 18])).all()

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])
print(torch.mm(a, b))
assert (torch.mm(a, b) == a @ b).all()

tensor = torch.tensor([1, 2, 3, 4])
assert tensor.sum() == 10

tensor = torch.tensor([[1, 2], [3, 4]])
assert (tensor.sum(dim=0) == torch.tensor([4, 6])).all()

tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
assert torch.isclose(tensor.mean(), torch.tensor(2.5))
assert torch.isclose(tensor.max(), torch.tensor(4.0))
assert torch.isclose(tensor.min(), torch.tensor(1.0))


a = torch.tensor([1, 2, 3])
b = torch.tensor([[1], [2], [3]])
print(a + b)

tensor = torch.tensor([1, 2, 3, 4, 5, 6])
reshaped_tensor = tensor.view(2, 3)
