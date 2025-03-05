
# n 个同学，至少两个人的生日相同
def func(n):
    none_prob = 1.0
    for i in range(n):
        none_prob *= (365 - i) / 365

    return 1 - none_prob

print(func(10))
