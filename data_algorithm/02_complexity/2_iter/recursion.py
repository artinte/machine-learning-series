
def recur(n):
    if n == 1:
        return 1
    return n + recur(n-1)


assert recur(100) == 5050


def fib(n):
    if n == 1 or n == 2:
        return n - 1

    return fib(n-1) + fib(n-2)


assert fib(5) == 3
