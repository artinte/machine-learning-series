def knapsack_dfs(weights, values, i, cap):
    # 若已选完所有物品或背包无剩余容量，则返回价值 0
    if i == 0 or cap == 0:
        return 0
    # 若超过背包容量，则只能选择不放入背包
    if weights[i - 1] > cap:
        return knapsack_dfs(weights, values, i - 1, cap)
    # 不放入
    no = knapsack_dfs(weights, values, i - 1, cap)
    # 放入物品
    yes = knapsack_dfs(weights, values, i - 1, cap - weights[i - 1]) + values[i - 1]
    return max(no, yes)


weights = [10, 20, 30, 40, 50]
values = [50, 120, 150, 210, 240]
capacity = 50
num = len(weights)
assert knapsack_dfs(weights, values, num, capacity) == 270
