import numpy
import sklearn.datasets
from matplotlib import pyplot
import sklearn.linear_model

def plot_decision_boundary(x, pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                            numpy.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(numpy.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    pyplot.contourf(xx, yy, Z, cmap='Wistia', alpha=0.8)
    pyplot.scatter(x[:, 0], x[:, 1], c=y)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

if __name__ == '__main__':
    rng = numpy.random.default_rng(0)
    X, y = sklearn.datasets.make_moons(200, noise=0.2)
    pyplot.scatter(X[:, 0], X[:, 1], s=40, c=y)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

    # Train the logistic regression classifier.
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    plot_decision_boundary(X, lambda x: clf.predict(x))

