import numpy
from matplotlib import pyplot

sigma = 2
duration = 10
dt = 0.01
num = int(duration / dt)

t = numpy.linspace(0, duration, num)
x = numpy.zeros(num)

rng = numpy.random.default_rng(0)
eta = rng.standard_normal(num)

for i in range(1, num):
    dot_x = sigma * eta[i]
    x[i] = x[i-1] + dot_x * dt

pyplot.plot(t, x)
pyplot.title('Simulated System with Gaussian White Noise')
pyplot.xlabel('Time (t)')
pyplot.ylabel('x(t)')
pyplot.grid(True)
pyplot.subplots_adjust(top=0.92, right=0.92)
pyplot.show()
