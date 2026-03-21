import os
import sys

# 1. 路径与命名空间注入 (必须置顶)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
THIRD_PARTY_DIR = os.path.join(BASE_DIR, 'third_party')

for d in [SRC_DIR, THIRD_PARTY_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

# 2. 镇压底层依赖 Bug
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

import logging
import argparse
import yaml
from yacs.config import CfgNode as CN

# 🌟 核心：直接导入我们的评估引擎
from umil.engine.evaluator import VideoEvaluator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# 动作词表保留在入口处，或者未来移入 data/processed_labels/
CANDIDATE_ACTIONS = [
    "cutting food", "spreading butter or jam", "making a sandwich", 
    # ... 请把你原来的 60 个 action 词表完整放在这里 ...
    "operating airplane controls", "looking out an airplane window", 
    "operating an excavator", "digging dirt with an excavator", "dumping dirt or rocks"
]

def main():
    parser = argparse.ArgumentParser(description="UMIL Evaluation Script")
    parser.add_argument('--config', required=True, help="Path to config yaml")
    parser.add_argument('--dataset', required=True, choices=['summe', 'tvsum'])
    parser.add_argument('--ckpt', required=True, help="Path to model weights")
    parser.add_argument('--test_keys', nargs='+', required=True, help='List of test video names')
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = CN(yaml.load(f, Loader=yaml.FullLoader))
        
    # 实例化引擎并执行
    evaluator = VideoEvaluator(
        config=config, 
        dataset_name=args.dataset, 
        checkpoint_path=args.ckpt, 
        candidate_actions=CANDIDATE_ACTIONS
    )
    
    evaluator.run(test_keys=args.test_keys)

if __name__ == '__main__':
    main()