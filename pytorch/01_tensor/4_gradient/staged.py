import math

x = 3 # example values
y = -4

# forward pass
sigy = 1.0 / (1 + math.exp(-y)) # sigmoid in numerator
num = x + sigy  # numerator
sigx = 1.0 / (1 + math.exp(-x)) # sigmoid in denominator
xpy = x + y
xpysqr = xpy**2
den = sigx + xpysqr # denominator
invden = 1.0 / den
f = num * invden # done!
assert round(f, 2) == 1.55

# backprop f = num * invden
dnum = invden   # gradient on numerator
dinvden = num
# backprop invden = 1.0 / den
dden = (-1.0 / (den**2)) * dinvden
# backprop den = sigx + xpysqr
dsigx = (1) * dden
dxpysqr = (1) * dden
# backprop xpysqr = xpy**2
dxpy = (2 * xpy) * dxpysqr
# backprop xpy = x + y
dx = (1) * dxpy
dy = (1) * dxpy
# backprop sigx = 1.0 / (1 + math.exp(-x))
dx += ((1 - sigx) * sigx) * dsigx # Notice += !! See notes below
# backprop num = x + sigy
dx += (1) * dnum
dsigy = (1) * dnum
# backprop sigy = 1.0 / (1 + math.exp(-y))
dy += ((1 - sigy) * sigy) * dsigy

assert round(dx, 2) == 2.06
assert round(dy, 2) == 1.59
