import numpy as np
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from umil.metrics.summary_protocol import generate_summary
from umil.metrics.knapsack import knapsack_dp

def test_knapsack_strict_capacity():
    """断言1：底层的动态规划 0-1 背包绝不能超重"""
    values = np.array([100, 200, 300])
    weights = np.array([10, 20, 30])
    capacity = 25
    
    selected = knapsack_dp(values, weights, capacity)
    total_weight = sum(weights[i] for i in selected)
    assert total_weight <= capacity, "🚨 背包算法算出的总重量超出了物理容量！"

def test_generate_summary_15percent_limit():
    """断言2：无论网络输出多么极端的高分，最终摘要总帧数绝不能超过 15%"""
    n_frames = 1000
    # 模拟 10 个均匀的镜头，每个镜头 100 帧
    cps = np.array([[i*100, (i+1)*100] for i in range(10)])
    nfps = [100] * 10
    positions = np.arange(1000)
    
    # 极端诱导：网络给所有帧都打了满分 1.0
    frame_scores = np.ones(1000)
    
    machine_summary = generate_summary(frame_scores, cps, n_frames, nfps, positions)
    
    assert len(machine_summary) == n_frames, "摘要数组长度与原视频帧数被破坏"
    
    selected_frames = np.sum(machine_summary)
    max_allowed = int(n_frames * 0.15)
    assert selected_frames <= max_allowed, \
        f"🚨 协议被破坏！选了 {selected_frames} 帧，超过了 15% 上限 ({max_allowed} 帧)"