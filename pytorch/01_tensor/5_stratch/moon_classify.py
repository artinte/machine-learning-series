import numpy
import sklearn.datasets
from matplotlib import pyplot

rng = numpy.random.default_rng(0)
x, y = sklearn.datasets.make_moons(200, noise=0.2)
pyplot.scatter(x[:, 0], x[:, 1], s=40, c=y)
pyplot.grid(True)
pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
pyplot.show()
