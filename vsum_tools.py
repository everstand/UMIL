import numpy as np
import math

def knapsack_dp(values, weights, capacity):
    """经典的 0-1 背包算法，用于在 15% 长度限制下挑选价值最大的镜头"""
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

def generate_summary(frame_scores, cps, n_frames, nfps, positions):
    """
    官方标准生成摘要序列：
    1. 把帧级别打分转化为镜头(Shot)级别打分
    2. 用背包算法挑选镜头 (限制总长度的 15%)
    3. 把选中的镜头还原为帧级别的 0/1 数组
    """
    # 1. 计算每个镜头(Shot)的平均分数
    shot_scores = []
    for i in range(len(cps)):
        start, end = cps[i]
        # 将原始帧索引映射到我们打分的下采样索引 (如果你的模型打分是原帧长，这里会自动对齐)
        start_p = np.searchsorted(positions, start)
        end_p = np.searchsorted(positions, end)
        if start_p == end_p:
            end_p += 1
        
        score = np.mean(frame_scores[start_p:end_p])
        shot_scores.append(score)

    # 为了适应背包算法，将浮点数分数放大转为整数
    shot_scores = np.array(shot_scores)
    shot_scores = (shot_scores * 100000).astype(int) 
    weights = nfps # 每个镜头的帧数
    capacity = int(n_frames * 0.15) # 顶会硬性规定：最大长度为总帧数的 15%

    # 2. 0-1 背包算法挑选镜头
    selected_shots = knapsack_dp(shot_scores, weights, capacity)

    # 3. 生成帧级别的 0/1 摘要
    machine_summary = np.zeros(n_frames, dtype=np.int32)
    for shot_idx in selected_shots:
        start, end = cps[shot_idx]
        machine_summary[start:end] = 1

    return machine_summary

def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """
    官方标准计算 F1-score：
    将机器生成的摘要，与多个真实人类的标注分别对比，取平均值(avg)或最大值(max)。
    SumMe 数据集通常取 Average F1。
    """
    machine_summary = np.asarray(machine_summary, dtype=np.int32)
    user_summary = np.asarray(user_summary, dtype=np.int32)
    _, n_frames = user_summary.shape

    # 确保长度一致
    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        pad = np.zeros(n_frames - len(machine_summary), dtype=np.int32)
        machine_summary = np.concatenate((machine_summary, pad))

    f1_scores = []
    precisions = []
    recalls = []

    # 与每一个人类标注者进行对比
    for user_idx in range(user_summary.shape[0]):
        gt_summary = user_summary[user_idx]
        overlap = np.sum(machine_summary * gt_summary)
        machine_sum = np.sum(machine_summary)
        user_sum = np.sum(gt_summary)

        p = overlap / machine_sum if machine_sum > 0 else 0
        r = overlap / user_sum if user_sum > 0 else 0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    if eval_metric == 'max':
        return np.max(f1_scores)
    else: # 'avg' 
        return np.mean(f1_scores)