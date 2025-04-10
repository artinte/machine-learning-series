import numpy
from matplotlib import pyplot
import torch
import show_heatmap

class NWKernelRegression(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = torch.nn.Parameter(torch.tensor([0.1]), requires_grad=True)

    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = torch.nn.functional.softmax(
            -((queries - keys) * self.w) ** 2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

def func(x):
    return 2 * numpy.sin(x) + x**0.8

if __name__ == '__main__':
    rng = numpy.random.default_rng(0)
    n_train = 50
    x_train = torch.tensor(numpy.sort(rng.random(n_train) * 5))
    assert x_train.shape == (50,)
    y_train = torch.tensor(func(x_train.numpy()) + rng.normal(0.0, 0.5, (n_train,)))
    queries = torch.tensor(numpy.arange(0, 5, 0.05))

    # x_tile = x_train.repeat((n_train, 1))
    # y_tile = y_train.repeat((n_train, 1))
    # 和除自己以外的所有训练样本的“键－值”对进行计算
    # keys = x_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # values = y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    # assert keys.shape == (50, 49)
    # assert values.shape == (50, 49)
    # 采用上面的 keys 和 values 方案，曲线会更加平滑些
    keys = x_train.repeat((n_train, 1))
    values = y_train.repeat((n_train, 1))
    assert keys.shape == (50, 50)
    assert values.shape == (50, 50)

    net = NWKernelRegression()
    loss = torch.nn.MSELoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.5)

    # 没有分批，全量训练 5 个轮回
    for epoch in range(5):
        trainer.zero_grad()
        l = loss(net(x_train, keys, values), y_train)
        l.sum().backward()
        trainer.step()
        print(f'Epoch {epoch + 1}, loss {float(l.mean().item()):.6f}')

    # 进行预测
    keys = x_train.repeat((len(queries), 1))
    values = y_train.repeat((len(queries), 1))
    y_hat = net(queries, keys, values).unsqueeze(1).detach()

    pyplot.plot(x_train, y_train, 'o', alpha=0.5, label='Samples')
    pyplot.plot(queries, func(queries.numpy()), label='Truth')
    pyplot.plot(queries, y_hat, label='Pred')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

    show_heatmap.show_heatmap(net.attention_weights.T.unsqueeze(0)
                              .unsqueeze(0).detach().numpy(),
                              'Sorted pred inputs',
                              'Sorted train inputs')
