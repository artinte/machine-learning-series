def coin_change_greedy(coins: list[int], amt) -> int:
    i = len(coins) - 1
    count = 0
    # 循环进行贪心选择，直到无剩余金额
    while amt > 0:
        while i > 0 and coins[i] > amt:
            i -= 1
        amt -= coins[i]
        count += 1
    # 若未找到可行方案，则返回 -1
    return count if amt == 0 else -1


coins = [1, 5, 10, 20, 50, 100]
print(coin_change_greedy(coins, 137))
