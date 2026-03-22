import os
import sys

# 1. 路径与命名空间注入
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
# 核心：使用完整的配置构建器，而非直接解析裸 YAML
from utils.config import get_config
from umil.engine.evaluator import VideoEvaluator

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="UMIL Evaluation Script")
    parser.add_argument('--config', required=True, help="Path to config yaml")
    parser.add_argument('--dataset', required=True, choices=['summe', 'tvsum'])
    parser.add_argument('--ckpt', required=True, help="Path to model weights")
    parser.add_argument('--test_keys', required=True, help='Path to the test split txt file')
    
    # 核心：暴露 yacs 动态配置覆写入口，供消融实验脚本驱动
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    
    # 构建带默认值与命令行覆写的完整配置树
    config = get_config(args)
    
    # 从物理文件加载测试键值
    if not os.path.exists(args.test_keys):
        raise FileNotFoundError(f"找不到测试集划分文件: {args.test_keys}")
    with open(args.test_keys, 'r') as f:
        test_keys = [line.strip() for line in f if line.strip()]
        
    # 实例化引擎
    # 动作词表的严格同源校验已在 VideoEvaluator 的 __init__ 中完成
    evaluator = VideoEvaluator(
        config=config, 
        dataset_name=args.dataset, 
        checkpoint_path=args.ckpt
    )
    
    evaluator.run(test_keys=test_keys)

if __name__ == '__main__':
    main()