import numpy
import sklearn.datasets
from matplotlib import pyplot
import sklearn.linear_model

def internal_decision_plot(x, pred_func):
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

def plot_single(x, pred_func):
    internal_decision_plot(x, pred_func)
    pyplot.grid(True)
    pyplot.scatter(x[:, 0], x[:, 1], c=y)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

def plot_multi(x, pred_func):
    internal_decision_plot(x, pred_func)
    pyplot.axis('off')
    pyplot.scatter(x[:, 0], x[:, 1], c=y, s=10)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.02)

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
    plot_single(X, lambda x: clf.predict(x))

    num_examples = len(X)   # training set size
    nn_input_dim = 2        # input layer dimensionlity
    nn_output_dim = 2       # output layer dimensionality

    # gradient descent parameters
    epsilon = 0.01          # learning rate fro gradient descent
    reg_lambda = 0.01       # regularization length

    # Helper function to evaluate the total loss on the dataset.
    def calculate_loss(model):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation to calculate our predictions.
        z1 = X.dot(W1) + b1
        a1 = numpy.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = numpy.exp(z2)
        probs = exp_scores / numpy.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        correct_logprobs = -numpy.log(probs[range(num_examples), y])
        data_loss = numpy.sum(correct_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += reg_lambda / 2 * (numpy.sum(numpy.square(W1)) + numpy.sum(numpy.square(W2)))
        return 1.0 / num_examples * data_loss

    def predict(model, x):
        W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
        # Forward propagation
        z1 = x.dot(W1) + b1
        a1 = numpy.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = numpy.exp(z2)
        probs = exp_scores / numpy.sum(exp_scores, axis=1, keepdims=True)
        return numpy.argmax(probs, axis=1)

    # This function learns parameters for the neural network and returns the model.
    # - nn_hdim: Number of nodes in the hidden layer
    # - num_passes: Number of passes through the training data for gradient descent
    # - print_loss: If True, print the loss every 1000 iterations
    def build_model(nn_hdim, num_passes=2000, print_loss=False):
        # Initialize the parameters to random values. We need to learn these.
        numpy.random.seed(0)
        W1 = numpy.random.randn(nn_input_dim, nn_hdim) / numpy.sqrt(nn_input_dim)
        b1 = numpy.zeros((1, nn_hdim))
        W2 = numpy.random.randn(nn_hdim, nn_output_dim) / numpy.sqrt(nn_hdim)
        b2 = numpy.zeros((1, nn_output_dim))

        # This is what we return at the end
        model = {}

        # Gradient descent. For each batch...
        for i in range(0, num_passes):
            # Forward propagation
            z1 = X.dot(W1) + b1
            a1 = numpy.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = numpy.exp(z2)
            probs = exp_scores / numpy.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(num_examples), y] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = numpy.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - numpy.power(a1, 2))
            dW1 = numpy.dot(X.T, delta2)
            db1 = numpy.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1

            # Gradient descent parameter update
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2

            # Assign new parameters to the model
            model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 100 == 0:
                print('Loss after iteration ' + str(i) + ': ' + str(round(calculate_loss(model), 4)))

        return model

    # Build a model with a 3-dimensional hidden layer
    model = build_model(3, print_loss=True)
    
    # Plot the decision boundary
    plot_single(X, lambda x: predict(model, x))
    pyplot.show()

    figure = pyplot.figure()
    hidden_layer_dims = [1, 2, 4, 10, 50, 100]
    for i , nn_dim in enumerate(hidden_layer_dims):
        figure.add_subplot(2, 3, i + 1)
        pyplot.title('Hidden size ' + str(nn_dim))
        model = build_model(nn_hdim=nn_dim)
        plot_multi(X, lambda x: predict(model, x))
    pyplot.show()
