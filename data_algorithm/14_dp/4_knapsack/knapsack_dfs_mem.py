def knapsack_dfs_mem(weights, values, mem, i, cap):
    if i == 0 or cap == 0:
        return 0
    if mem[i][cap] != -1:
        return mem[i][cap]
    
    if cap - weights[i - 1] < 0:
        result = knapsack_dfs_mem(weights, values, mem, i - 1, cap)
        mem[i][cap] = result
        return result
    
    no = knapsack_dfs_mem(weights, values, mem, i - 1, cap)
    yes = knapsack_dfs_mem(weights, values, mem,  i - 1, cap - weights[i - 1]) + values[i - 1]
    result = max(no, yes)
    mem[i][cap] = result
    return result


weights = [10, 20, 30, 40, 50]
values = [50, 120, 150, 210, 240]
capacity = 50
num = len(weights)
mem = [[-1] * (capacity + 1) for _ in range(num + 1)]
assert knapsack_dfs_mem(weights, values, mem, num, capacity) == 270
