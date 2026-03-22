import os
import sys
import logging
import argparse
import yaml
import re

# 1. 路径注入
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
THIRD_PARTY_DIR = os.path.join(BASE_DIR, 'third_party')

for d in [SRC_DIR, THIRD_PARTY_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

from utils.config import get_config
from umil.engine.evaluator import VideoEvaluator
from umil.datasets.splits import load_split

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="UMIL Evaluation Script")
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True, choices=['summe', 'tvsum'])
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--test_keys', required=True, help='Path to tvsum.yml')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    config = get_config(args)
    
    # 🌟 自动识别 split 索引
    split_idx = 0
    match = re.search(r'split(\d+)', args.ckpt)
    if match:
        split_idx = int(match.group(1))
        
    # 🌟 统一调用 splits.py，直接拿到纯净的 H5 Key 列表
    _, test_keys = load_split(args.test_keys, split_idx)
    
    logger.info(f"成功加载 Split-{split_idx}，共 {len(test_keys)} 个测试视频。")
        
    evaluator = VideoEvaluator(config, args.dataset, args.ckpt)

if __name__ == '__main__':
    main()