import numpy as np

# forward pass
W = np.random.randn(5, 10)
X = np.random.randn(10, 3)
D = W.dot(X)

# now suppose we had the gradient on D from above in the circuit
dD = np.random.randn(*D.shape)  # same shape as D
dW = dD.dot(X.T)    # .T gives the transpose of the matrix
dX = W.T.dot(dD)
