import os
import sys
from umil.datasets.metadata.tvsum_metadata import TVSUM_STATIC_MAP

# 临时将根目录加入路径，读取你目前在根目录的映射文件
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from umil.datasets.metadata.tvsum_metadata import TVSUM_STATIC_MAP

def test_tvsum_map_completeness():
    """断言1：TVSum 必须严格包含 50 个视频的映射"""
    assert len(TVSUM_STATIC_MAP) == 50, \
        f"🚨 致命错误：映射表长度应为 50，当前为 {len(TVSUM_STATIC_MAP)}"

def test_tvsum_map_uniqueness():
    """断言2：映射表中的物理视频绝对不能有重复（无碰撞）"""
    all_videos = list(TVSUM_STATIC_MAP.values())
    unique_videos = set(all_videos)
    assert len(all_videos) == len(unique_videos), \
        "🚨 致命错误：TVSUM_STATIC_MAP 中存在重复分配的物理视频！"

def test_tvsum_video_existence():
    """断言3：如果物理文件夹存在，映射表中的视频必须 100% 能找到"""
    video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/TVSum/videos'))
    if not os.path.exists(video_dir):
        return  # 允许在无数据的 CI 服务器上静默跳过物理检查
        
    for k, vname in TVSUM_STATIC_MAP.items():
        vpath = os.path.join(video_dir, f"{vname}.mp4")
        assert os.path.exists(vpath), f"🚨 找不到物理视频文件: {vpath}"