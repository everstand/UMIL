import math
import numpy as np
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from umil.metrics.fscore import evaluate_summary

def test_fscore_perfect_overlap():
    """断言1：机器预测与人类完全重合时，F1 为 1.0"""
    machine = np.array([1, 1, 0, 0, 1])
    user = np.array([[1, 1, 0, 0, 1]]) # 单个标注者
    
    f1 = evaluate_summary(machine, user, eval_metric='max')
    assert math.isclose(f1, 1.0), "完美重合时 F1 应为 1.0"

def test_fscore_zero_overlap():
    """断言2：机器预测与人类完全错开时，F1 为 0.0"""
    machine = np.array([1, 1, 0, 0])
    user = np.array([[0, 0, 1, 1]])
    
    f1 = evaluate_summary(machine, user, eval_metric='avg')
    assert f1 == 0.0, "无重合时 F1 应为 0.0"

def test_fscore_multiple_annotators_avg():
    """断言3：TVSum 协议下，多个标注者的 F1 必须求平均 (Average F1)"""
    machine = np.array([1, 0])
    # 标注者 A 完美命中 (F1=1.0)，标注者 B 完全错开 (F1=0.0)
    user = np.array([[1, 0], [0, 1]])
    
    f1_avg = evaluate_summary(machine, user, eval_metric='avg')
    assert math.isclose(f1_avg, 0.5), "TVSum 的多标注者 F1 没有正确求平均值"