import numpy as np

def get_summ_diversity(machine_summary, features):
    """
    计算多样性：通过所选特征间的平均余弦距离衡量。
    契约约束：摘要数组必须与物理特征在时间维度上绝对等长！
    """
    machine_summary = np.asarray(machine_summary, dtype=np.bool_)
    
    # 🌟 剥离隐式容错，替换为极其暴力的防御断言 (Fail-Fast)
    assert len(machine_summary) == len(features), \
        f"协议层崩溃：机器摘要长度 ({len(machine_summary)}) 与特征序列长度 ({len(features)}) 物理不对齐！请在 Adapter 层排查数据对齐逻辑！"
        
    selected_features = features[machine_summary]
    if len(selected_features) < 2:
        return 0.0
        
    norm = np.linalg.norm(selected_features, axis=1, keepdims=True)
    norm[norm == 0] = 1e-10
    selected_features = selected_features / norm
    
    sim_matrix = np.dot(selected_features, selected_features.T)
    div_matrix = 1.0 - sim_matrix
    
    n = len(selected_features)
    diversity = (np.sum(div_matrix) - np.trace(div_matrix)) / (n * (n - 1))
    
    return float(diversity)