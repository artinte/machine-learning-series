import numpy
from matplotlib import pyplot

rng = numpy.random.default_rng(seed=0)
random_numbers = rng.standard_normal(size=100)
input = numpy.linspace(0, 4, 100)
y_true = numpy.round(3 * input + 4 + random_numbers, 4)

pyplot.scatter(input, y_true)
pyplot.grid(True)
pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
pyplot.show()

param = 1
step = 0.01
epochs = 5
intermediate_list = [1.0]
for _ in range(epochs):
    for index, x in enumerate(input):
        y_pred = numpy.round(param * x + 4, 4)
        if y_pred < y_true[index]:
            param += step
        else:
            param -= step
        intermediate_list.append(param)
print('Last param is', round(intermediate_list[-1], 4))

pyplot.scatter(input, y_true)
pyplot.grid(True)
pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
pyplot.plot(input, intermediate_list[-1] * input + 4, color='red')
pyplot.show()

pyplot.plot(range(len(intermediate_list)), intermediate_list)
pyplot.grid(True)
pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
pyplot.show()