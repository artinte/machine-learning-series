import torch

if __name__ == '__main__':
    x = torch.ones((2, 1, 4))
    y = torch.ones((2, 4, 6))
    print(torch.bmm(x, y).shape)

    weights = (torch.ones((2, 10)) * 0.1).unsqueeze(1)
    # 末尾插入一个维度
    values = torch.arange(20.0).reshape((2, 10)).unsqueeze(-1)
    print('Weights shape:', weights.shape)
    print('Values shape:', values.shape)
    result = torch.bmm(weights, values)
    print('Result shape:', result.shape)
    print(result)

