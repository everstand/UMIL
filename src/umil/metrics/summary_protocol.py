import numpy as np
from .knapsack import knapsack_dp

def generate_summary(frame_scores, cps, n_frames, nfps, positions):
    """
    兼容三种输入：
    1) frame_scores 长度 == n_frames                     -> full-frame 分数
    2) frame_scores 长度 ~= n_frames (差 1~2 帧)         -> 视为 metadata 偏差，自动 pad/trim
    3) frame_scores 长度 == len(positions)               -> pick-level 分数，先展开回 full-frame
    """
    frame_scores = np.asarray(frame_scores, dtype=np.float32).reshape(-1)
    cps = np.asarray(cps, dtype=np.int64)
    positions = np.asarray(positions, dtype=np.int64).reshape(-1)

    if n_frames <= 0:
        return np.zeros(0, dtype=np.int32)

    positions = positions[positions < n_frames]
    if len(positions) == 0:
        raise ValueError("positions is empty after clipping to n_frames.")

    # -------- Step 1: 统一成 full-frame scores --------
    # 情况 A：full-frame，长度精确匹配
    if len(frame_scores) == n_frames:
        full_frame_scores = frame_scores.copy()

    # 情况 B：full-frame，但和 metadata 有轻微 off-by-one / off-by-two
    elif abs(len(frame_scores) - n_frames) <= 2:
        full_frame_scores = np.zeros(n_frames, dtype=np.float32)
        L = min(len(frame_scores), n_frames)
        full_frame_scores[:L] = frame_scores[:L]
        if L < n_frames:
            fill_val = frame_scores[-1] if len(frame_scores) > 0 else 0.0
            full_frame_scores[L:] = fill_val

    # 情况 C：pick-level 分数，需要展开到 full-frame
    elif len(frame_scores) == len(positions):
        full_frame_scores = np.zeros(n_frames, dtype=np.float32)

        expanded_positions = positions
        if expanded_positions[-1] != n_frames:
            expanded_positions = np.concatenate([expanded_positions, [n_frames]])

        L = min(len(frame_scores), len(expanded_positions) - 1)
        for i in range(L):
            left = int(expanded_positions[i])
            right = int(expanded_positions[i + 1])
            if right > left:
                full_frame_scores[left:right] = frame_scores[i]

    else:
        raise ValueError(
            f"Length mismatch: len(frame_scores)={len(frame_scores)}, "
            f"n_frames={n_frames}, len(positions)={len(positions)}"
        )

    # -------- Step 2: frame-level -> shot-level --------
    shot_scores = []
    for start, end in cps:
        start = max(0, int(start))
        end = min(n_frames - 1, int(end))
        if end < start:
            shot_scores.append(0.0)
            continue
        shot_scores.append(float(np.mean(full_frame_scores[start:end + 1])))

    shot_scores = np.asarray(shot_scores, dtype=np.float32)
    values = (shot_scores * 100000).astype(np.int64)
    weights = list(map(int, nfps))
    capacity = int(n_frames * 0.15)

    # -------- Step 3: knapsack --------
    selected_shots = knapsack_dp(values, weights, capacity)

    # -------- Step 4: shot-level -> binary summary --------
    machine_summary = np.zeros(n_frames, dtype=np.int32)
    for shot_idx in selected_shots:
        start, end = cps[shot_idx]
        start = max(0, int(start))
        end = min(n_frames - 1, int(end))
        if end >= start:
            machine_summary[start:end + 1] = 1

    return machine_summary