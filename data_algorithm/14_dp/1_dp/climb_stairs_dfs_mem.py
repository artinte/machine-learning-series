def climb_stairs_dfs_mem(n, mem):
    if n == 1 or n == 2:
        return n
    if mem[n] != -1:
        return mem[n]
    count = climb_stairs_dfs_mem(n - 1, mem) + climb_stairs_dfs_mem(n - 2, mem)
    mem[n] = count
    return count


n = 20
# mem[i] 记录爬到第 i阶的方案总数，-1 代表无记录
mem = [-1] * (n + 1)
print(climb_stairs_dfs_mem(n, mem))
