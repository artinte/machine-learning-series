def sum(n):
    result = 0
    # 循环求和 1, 2, ... , n-1, n
    for i in range(1, n+1):
        result += i
    return result


assert sum(100) == 5050
