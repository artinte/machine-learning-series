import numpy
from matplotlib import pyplot

if __name__ == '__main__':
    x_values = [0, 5]
    y_values = [250, 665]
    x = numpy.linspace(0, 5, 100)
    y = 83 * x + 250
    for i, txt in enumerate(zip(x_values, y_values)):
        pyplot.annotate(f'{txt}', (x_values[i], y_values[i]), textcoords="offset points", xytext=(0, 10))
    pyplot.text(2.5, 400, 'D(t) = 83t + 250', fontsize=12, color='blue')
    pyplot.scatter(x_values, y_values, color='red')
    pyplot.plot(x, y)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
