import numpy

if __name__ == '__main__':
    a = numpy.array([1, 2, 3, 4, 5, 6])
    print(a, a[0])
    a[0] = 10
    print(a, a[0], a[:3])
    b = a[3:]
    print(b)
    b[0] = 40
    print(a, b)

