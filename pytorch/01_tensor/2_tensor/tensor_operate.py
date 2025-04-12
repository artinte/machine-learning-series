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
c = a + b
assert a.shape == (3,)
assert b.shape == (3, 1)
assert c.shape == (3, 3)
print(c)

x = torch.randn((2, 3, 4))
y = x.view((6, 4))
assert x.shape == (2, 3, 4)
assert y.shape == (6, 4)

z = x.view(-1, 4)
assert z.shape == (6, 4)

x = torch.randn((2, 3, 4))
y = x.reshape((6, 4))
assert x.data_ptr() == y.data_ptr()

x = torch.randn((2, 3))
y = x.transpose(0, 1)
assert y.shape == (3, 2)

x = torch.randn(2, 3, 4)
y = x.transpose(1, 2)
assert y.shape == (2, 4, 3)

x = torch.randn((1, 3, 1, 4))
y = x.squeeze()
assert y.shape == (3, 4)

x = torch.randn((1, 3, 1, 4))
y = x.squeeze(0)
assert y.shape == (3, 1, 4)

x = torch.randn((3, 4))
y = x.unsqueeze(0)
assert y.shape == (1, 3, 4)

x = torch.randn((2, 3, 4))
y = x.permute(2, 0, 1)
assert y.shape == (4, 2, 3)

x = torch.randn((2, 3, 4))
y = x.flatten()
assert y.shape == (24,)

y = x.flatten(start_dim=1)
assert y.shape == (2, 12)
