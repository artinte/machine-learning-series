def climb_stairs_dfs(n):
    if n == 1 or n == 2:
        return n
    count = climb_stairs_dfs(n - 1) + climb_stairs_dfs(n - 2)
    return count


print(climb_stairs_dfs(20))
