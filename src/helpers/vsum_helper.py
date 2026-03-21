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
        # 将原始帧索引映射到打分的下采样索引
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
    capacity = int(n_frames * 0.15) # 最大长度为总帧数的 15%

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

    # 与每一个人类标注者进行对比
    for user_idx in range(user_summary.shape[0]):
        gt_summary = user_summary[user_idx]
        overlap = np.sum(machine_summary * gt_summary)
        machine_sum = np.sum(machine_summary)
        user_sum = np.sum(gt_summary)

        p = overlap / machine_sum if machine_sum > 0 else 0
        r = overlap / user_sum if user_sum > 0 else 0
        f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0

        f1_scores.append(f1)

    if eval_metric == 'max':
        return np.max(f1_scores)
    else: # 'avg' 
        return np.mean(f1_scores)

def get_summ_diversity(machine_summary, features):
    """
    DSNet / VASNet 官方标配：多样性 (Diversity) 算法。
    衡量摘要中被选中镜头的视觉特征差异度。
    计算逻辑： 1 - 选中特征之间的平均余弦相似度。
    """
    machine_summary = np.asarray(machine_summary, dtype=np.int32)
    n_features = features.shape[0]
    n_frames = len(machine_summary)
    
    # 🌟 降维打击与对齐保护：
    # machine_summary 是原视频帧总长度，而 features 是 H5 里下采样过的。必须抽样对齐。
    if n_frames != n_features:
        indices = np.linspace(0, n_frames - 1, n_features).astype(int)
        selected_flags = machine_summary[indices]
    else:
        selected_flags = machine_summary
        
    # 拿到所有被选中帧的索引
    selected_indices = np.where(selected_flags > 0)[0]
    
    # 如果选中的帧少于 2 帧，无法计算成对相似度，属于极端退化情况
    if len(selected_indices) < 2:
        return 0.0
        
    # 提取特征并做 L2 归一化
    sel_features = features[selected_indices]
    norm = np.linalg.norm(sel_features, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10  # 防止死机报错
    sel_features = sel_features / norm
    
    # 计算 NxN 的余弦相似度矩阵
    sim_matrix = np.dot(sel_features, sel_features.T)
    
    # 提取上三角矩阵的非对角线元素 (剔除自身相似度 1.0，避免双向重复计算)
    n = len(selected_indices)
    upper_tri_idx = np.triu_indices(n, k=1)
    
    if len(upper_tri_idx[0]) == 0:
        return 0.0
        
    avg_sim = np.mean(sim_matrix[upper_tri_idx])
    
    # 多样性 = 1 - 平均相似度
    diversity = 1.0 - avg_sim
    
    return float(diversity)

def get_summ_diversity(machine_summary, features):
    """
    官方标准计算 Diversity (多样性)：
    通过计算所选帧(或镜头)特征之间的平均余弦距离 (1 - Cosine Similarity) 来衡量。
    """
    machine_summary = np.asarray(machine_summary, dtype=np.bool_)
    
    # 核心工程防御：长度对齐 (Alignment)
    if len(machine_summary) != len(features):
        indices = np.linspace(0, len(machine_summary) - 1, len(features)).astype(int)
        machine_summary = machine_summary[indices]
        
    # 提取被选中的有效特征
    selected_features = features[machine_summary]
    
    # 如果选中的特征少于2个，无法计算两两差异，多样性为 0
    if len(selected_features) < 2:
        return 0.0
        
    # 1. 归一化特征 (L2 Norm) 
    norm = np.linalg.norm(selected_features, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    selected_features = selected_features / norm
    
    # 2. 计算两两之间的余弦相似度矩阵
    sim_matrix = np.dot(selected_features, selected_features.T)
    
    # 3. 多样性矩阵 = 1 - 相似度矩阵
    div_matrix = 1.0 - sim_matrix
    
    # 4. 剔除对角线并计算均值
    n = len(selected_features)
    diversity = (np.sum(div_matrix) - np.trace(div_matrix)) / (n * (n - 1))
    
    return float(diversity)