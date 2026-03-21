import numpy as np

def knapsack_dp(values, weights, capacity):
    """经典的 0-1 背包算法，严格限制在指定 capacity 下挑选价值最大的物品"""
    n = len(values)
    dp = np.zeros((n + 1, capacity + 1), dtype=np.int32)
    
    for i in range(1, n + 1):
        w = weights[i - 1]
        v = values[i - 1]
        for c in range(1, capacity + 1):
            if w <= c:
                dp[i][c] = max(dp[i - 1][c], dp[i - 1][c - w] + v)
            else:
                dp[i][c] = dp[i - 1][c]

    selected = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            selected.append(i - 1)
            c -= weights[i - 1]
    return selected[::-1]