import numpy
from matplotlib import pyplot
import torch
import show_heatmap

def func(x):
    return 2 * numpy.sin(x) + x**0.8

if __name__ == '__main__':
    rng = numpy.random.default_rng(0)
    n_train = 50
    x_train = numpy.sort(rng.random(n_train) * 5)
    y_train = func(x_train) + rng.normal(0.0, 0.5, (n_train,))
    x = numpy.arange(0, 5, 0.05)
    y_truth = func(x)

    # Each row contains the same input (query).
    x_pred_repeat = x.repeat(n_train).reshape((-1, n_train))
    assert x_pred_repeat.shape == (100, 50)
    # shape: (n_pred, n_train)
    attention_weights = torch.nn.functional.softmax(-(torch.tensor(x_pred_repeat - x_train))**2 / 2, dim=1)
    print('Attention weigths shape:', attention_weights.shape)
    assert attention_weights.shape == (100, 50)
    assert y_train.shape == (50,)
    # 计算得到 100 个预测值
    y_hat = torch.matmul(attention_weights, torch.tensor(y_train))
    assert y_hat.shape == (100,)
    pyplot.plot(x_train, y_train, 'o', alpha=0.5, label='Samples')
    pyplot.plot(x, y_truth, label='Truth')
    pyplot.plot(x, y_hat, label='Pred')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

    show_heatmap.show_heatmap(attention_weights.T.unsqueeze(0)
                              .unsqueeze(0).detach().numpy(),
                              'Sorted pred inputs',
                              'Sorted train inputs')
