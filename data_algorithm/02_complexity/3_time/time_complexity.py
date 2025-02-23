
arr = [10, 20, 30, 40]
assert arr[2] == 30  # O(1)


def linear(n):
    count = 0
    for _ in range(n):
        count += 1
    return count


assert linear(10) == 10


def find_duplicates(arr):
    n = len(arr)
    duplicates = set()
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] == arr[j]:
                duplicates.add(arr[i])
    return list(duplicates)


assert find_duplicates([1, 2, 3, 2, 4, 1, 5]) == [1, 2]


def generate_subsets(arr, index=0, subset=[]):
    if index == len(arr):
        print(subset)
        return
    # 不选当前元素
    generate_subsets(arr, index + 1, subset)
    # 选当前元素
    generate_subsets(arr, index + 1, subset + [arr[index]])


generate_subsets([1, 2, 3])


def gcd(a, b):
    # 使用欧几里得算法求最大公约数
    while b:
        a, b = b, a % b
    return a


assert (gcd(48, 18)) == 6
