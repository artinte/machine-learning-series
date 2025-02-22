def backtrack(choices: list[int], state, total, result):
    # 当爬到第 n 阶时，方案数量加 1
    if state == total:
        result[0] += 1
    # 遍历所有选择
    for choice in choices:
        # 剪枝：不允许越过第 top 阶
        if state + choice > total:
            continue
        # 尝试：做出选择，更新状态
        backtrack(choices, state + choice, total, result)


def climbing_stairs_backtrack(total):
    # 可选择向上爬 1 阶或者 2 阶
    choices = [1, 2]
    # 从第 0 阶开始爬
    state = 0
    # 记录方案数量
    result = [0]
    backtrack(choices, state, total, result)
    return result[0]


print(climbing_stairs_backtrack(20))
