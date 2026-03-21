import numpy as np

def evaluate_summary(machine_summary, user_summary, eval_metric='avg'):
    """计算 F1-score：SumMe 用 max，TVSum 用 avg"""
    machine_summary = np.asarray(machine_summary, dtype=np.int32)
    user_summary = np.asarray(user_summary, dtype=np.int32)
    _, n_frames = user_summary.shape

    if len(machine_summary) > n_frames:
        machine_summary = machine_summary[:n_frames]
    elif len(machine_summary) < n_frames:
        pad = np.zeros(n_frames - len(machine_summary), dtype=np.int32)
        machine_summary = np.concatenate((machine_summary, pad))

    f1_scores = []
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
        return float(np.max(f1_scores))
    else: 
        return float(np.mean(f1_scores))