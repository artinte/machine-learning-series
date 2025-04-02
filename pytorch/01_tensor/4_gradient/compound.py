# set some inputs
x = -2; y = 5; z = -4

# perform the forward pass
q = x + y   # q becomes 3
f = q * z   # f becomes -12

# perform the backward pass (backpropagation) in reverse order:
# first backprop through f = q * z
df_dz = q   # df/dz = q, so gradient on z becomes 3
df_dq = z   # df/dq = z, so gradient on q becomes -4
dq_dx = 1.0
dq_dy = 1.0

# now backprop through q = x + y
df_dx = df_dq * dq_dx   # the multiplication here is the chain rule
df_dy = df_dq * dq_dy

assert df_dx == -4
assert df_dy == -4
assert df_dz == 3
