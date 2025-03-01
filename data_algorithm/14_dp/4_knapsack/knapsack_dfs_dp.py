def knapsack_dfs_dp(weights, values, cap):
    n = len(weights)
    dp = [[0] * (cap + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, cap + 1):
            if weights[i - 1] > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - weights[i - 1]] + values[i - 1])
    return dp[n][cap]


weights = [10, 20, 30, 40, 50]
values = [50, 120, 150, 210, 240]
capacity = 50
num = len(weights)
assert knapsack_dfs_dp(weights, values, capacity) == 270
