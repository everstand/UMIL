import numpy as np
from .knapsack import knapsack_dp

def generate_summary(frame_scores, cps, n_frames, nfps, positions):
    """
    官方标准生成摘要序列：
    1. 帧得分转镜头得分
    2. 背包算法挑选 (严格 15% 容量限制)
    3. 还原为 0/1 数组
    """
    shot_scores = []
    for i in range(len(cps)):
        start, end = cps[i]
        start_p = np.searchsorted(positions, start)
        end_p = np.searchsorted(positions, end)
        if start_p == end_p:
            end_p += 1
        score = np.mean(frame_scores[start_p:end_p])
        shot_scores.append(score)

    shot_scores = np.array(shot_scores)
    shot_scores = (shot_scores * 100000).astype(int) 
    weights = nfps 
    capacity = int(n_frames * 0.15) 

    selected_shots = knapsack_dp(shot_scores, weights, capacity)

    machine_summary = np.zeros(n_frames, dtype=np.int32)
    for shot_idx in selected_shots:
        start, end = cps[shot_idx]
        machine_summary[start:end] = 1

    return machine_summary