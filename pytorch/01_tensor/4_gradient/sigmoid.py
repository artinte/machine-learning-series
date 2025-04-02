import math

w = [2, -3, -3] # assume some random weights and data
x = [-1, -2]

# forward pass
dot = w[0] * x[0] + w[1] * x[1] + w[2]
f = 1.0 / (1 + math.exp(-dot))

# backward pass through the neuron (backpropagation)
ddot = (1 - f) * f  # gradient on dot variable, using the sigmoid gradient derivation
dx = [w[0] * ddot, w[1] * ddot] # backprop into x
dw = [x[0] * ddot, x[1] * ddot, 1.0 * ddot]

print([round(x, 2) for x in dx])
print([round(w, 2) for w in dw])
