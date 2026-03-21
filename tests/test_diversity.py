import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from umil.metrics.diversity import get_summ_diversity


def test_diversity_all_zeros():
    """断言1：当模型什么都没选时，Diversity 必须为 0，且不抛出除以零报错"""
    machine_summary = np.zeros(100)
    features = np.random.randn(100, 1024) 
    
    div = get_summ_diversity(machine_summary, features)
    assert div == 0.0, "空摘要的多样性应为 0.0"

def test_diversity_identical_features():
    """断言2：如果选中的帧视觉特征完全一样，Diversity 必须为 0"""
    machine_summary = np.ones(50)
    identical_feat = np.ones(1024)
    features = np.array([identical_feat for _ in range(50)])
    
    div = get_summ_diversity(machine_summary, features)
    assert np.isclose(div, 0.0, atol=1e-5), "完全相同画面的多样性应为 0.0"

def test_diversity_orthogonal_features():
    """断言3：如果选中的帧特征两两完全正交（完全不相关），Diversity 应趋近于 1.0"""
    machine_summary = np.array([1, 1])
    features = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    div = get_summ_diversity(machine_summary, features)
    assert np.isclose(div, 1.0, atol=1e-5), "正交特征的多样性应为 1.0"