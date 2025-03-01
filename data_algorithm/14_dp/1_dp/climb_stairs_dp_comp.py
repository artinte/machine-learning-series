def climb_stairs_dp_comp(n):
    if n == 1 or n == 2:
        return n
    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b
    return b


assert climb_stairs_dp_comp(20) == 10946
