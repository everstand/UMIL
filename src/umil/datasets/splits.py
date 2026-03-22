import yaml
import logging

logger = logging.getLogger(__name__)

def load_split(split_file, split_id):
    """
    统一的数据划分解析器。
    无论什么数据集，一律返回纯净的 H5 内部键 (如 ['video_1', 'video_2'])。
    """
    with open(split_file, 'r') as f:
        splits = yaml.load(f, Loader=yaml.FullLoader)
        
    target_data = splits[split_id]
    
    # 穿透 YAML 特有的列表嵌套
    if isinstance(target_data, list):
        target_data = target_data[0]
        
    if not isinstance(target_data, dict):
        raise TypeError(f"预期 Split 数据为字典格式，但得到的是 {type(target_data)}")
        
    raw_train = target_data.get('train_keys', [])
    raw_test = target_data.get('test_keys', [])
    
    # 清洗掉 '../datasets/.../' 前缀，实行 H5 Key 霸权
    train_h5_keys = [p.split('/')[-1] for p in raw_train]
    test_h5_keys = [p.split('/')[-1] for p in raw_test]
    
    logger.info(f"==> 成功加载 Split-{split_id} | Train: {len(train_h5_keys)} 视频 | Test: {len(test_h5_keys)} 视频")
    
    return train_h5_keys, test_h5_keys