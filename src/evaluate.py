import os
import sys

# ================= 解决重构后的路径问题 =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
# ========================================================

import argparse
import h5py
import torch
import numpy as np
import decord
from tqdm import tqdm
import yaml
from yacs.config import CfgNode as CN
import clip

from models.xclip import build_model
from helpers.vsum_helper import generate_summary, evaluate_summary

# 全局变量：候选动作词表
CANDIDATE_ACTIONS = [
    "cutting food", "spreading butter or jam", "making a sandwich", 
    "pouring liquid", "washing hands or dishes", "chewing and swallowing food",
    "using hand tools", "operating a car jack", "tightening or loosening tire nuts", 
    "handling heavy tires", "pushing a stuck car", "digging dirt or snow", 
    "attaching a tow rope", "driving a car", "stepping on car pedals", "waiting in a car",
    "grooming a pet", "washing a pet with water", "applying pet shampoo", 
    "drying a pet with a towel", "petting an animal", "walking a dog on a leash", 
    "feeding a pet", "a dog jumping through a hoop", "a dog sitting on command",
    "running or sprinting", "parkour jumping over obstacles", "climbing a wall", 
    "rolling on the ground", "riding a BMX bike", "balancing on a bike", 
    "falling down", "deploying a parachute", "wearing extreme sports gear", 
    "scuba diving underwater", "swimming", "aiming and shooting a paintball gun", 
    "swinging a mallet in bike polo", "sliding down a water slide",
    "marching in a parade", "dancing in a flash mob", "playing a musical instrument", 
    "waving to a crowd", "holding a flag or banner", "clapping and cheering", 
    "standing in a crowd", "talking to someone", "setting up dominoes", "lighting a fire",
    "walking around a tourist attraction", "looking up at a monument", 
    "taking a picture or recording a video", "posing for a photo", 
    "children playing and chasing", "climbing playground equipment",
    "operating airplane controls", "looking out an airplane window", 
    "operating an excavator", "digging dirt with an excavator", "dumping dirt or rocks"
]

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    (完全对标 dsnet 和 PyTorch 官方推荐的计量工具)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def build_fingerprint_db(video_dir):
    """将指纹匹配逻辑抽离为独立函数，保持主函数整洁"""
    fingerprints = {}
    if os.path.exists(video_dir):
        for fname in os.listdir(video_dir):
            if fname.endswith(('.mp4', '.avi', '.webm', '.mkv')):
                vpath = os.path.join(video_dir, fname)
                try:
                    fingerprints[vpath] = len(decord.VideoReader(vpath))
                except Exception:
                    pass
    return fingerprints


def predict_video_scores(model, video_path, text_tokens, device, config):
    """
    负责视频流读取与模型推理 (前向传播)
    返回: np.array (frame_scores)
    """
    vr = decord.VideoReader(video_path, width=224, height=224)
    total_frames = len(vr)
    
    clip_len = config.DATA.NUM_FRAMES
    frame_interval = config.DATA.FRAME_INTERVAL
    actual_clip_len = clip_len * frame_interval

    frame_scores = []
    for start_idx in range(0, total_frames, actual_clip_len):
        end_idx = min(start_idx + actual_clip_len, total_frames)
        actual_len = end_idx - start_idx
        frame_indices = list(range(start_idx, end_idx, frame_interval))
        
        if len(frame_indices) < clip_len:
            frame_indices.extend([frame_indices[-1]] * (clip_len - len(frame_indices)))
            
        frames = vr.get_batch(frame_indices).asnumpy()
        frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0).to(device)
        frames_tensor = frames_tensor / 255.0
        
        with torch.no_grad():
            outputs = model(frames_tensor, text_tokens)
            
        # 智能字典拆包
        if isinstance(outputs, dict):
            logits = outputs.get('predicts', outputs.get('logits', outputs.get('output', list(outputs.values())[0])))
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        score = torch.max(torch.sigmoid(logits), dim=1)[0].item()
        frame_scores.extend([score] * actual_len)

    return np.array(frame_scores)


def evaluate(config, dataset_name, checkpoint_path, test_keys=None):
    """
    主评估流水线 (对标 dsnet evaluate.py)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. 动态路由
    if dataset_name == 'summe':
        h5_path = "data/eccv16_datasets/eccv16_dataset_summe_google_pool5.h5"
        video_dir = "data/SumMe/videos"
    elif dataset_name == 'tvsum':
        h5_path = "data/eccv16_datasets/eccv16_dataset_tvsum_google_pool5.h5"
        video_dir = "data/TVSum/videos"
    else:
        raise ValueError(f"🚨 未知数据集: {dataset_name}")

    # 2. 加载模型
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"🚨 找不到权重文件: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    import logging
    logger = logging.getLogger(__name__)
    model = build_model(state_dict=new_state_dict, T=config.DATA.NUM_FRAMES, logger=logger).to(device)
    model.eval()

    # 3. 准备评测依赖
    text_prompts = [f"A video of a person {action}" for action in CANDIDATE_ACTIONS]
    text_tokens = clip.tokenize(text_prompts).to(device)
    fingerprints = build_fingerprint_db(video_dir)
    
    # 引入 dsnet 经典的算分器
    f1_meter = AverageMeter()

    # 4. 执行评估循环
    print(f"📂 正在加载官方 H5 标签库: {h5_path}")
    with h5py.File(h5_path, 'r') as h5_data:
        keys = list(h5_data.keys())
        
        for key in tqdm(keys, desc=f"Evaluating {dataset_name.upper()}", ncols=100):
            if test_keys is not None and key not in test_keys:
                continue
            try:
                n_frames_h5 = h5_data[key]['n_frames'][()]
                video_path = None
                
                # --- A. 匹配视频文件 ---
                if 'video_name' in h5_data[key]:
                    vid_name = h5_data[key]['video_name'][()]
                    vid_name = vid_name.item().decode('utf-8') if hasattr(vid_name, 'item') else str(vid_name.decode('utf-8'))
                    for ext in ['.mp4', '.avi', '.webm', '.mkv']:
                        tmp_path = os.path.join(video_dir, f"{vid_name}{ext}")
                        if os.path.exists(tmp_path):
                            video_path = tmp_path
                            break

                # 盲匹配兜底
                if not video_path:
                    min_diff = float('inf')
                    for vpath, vlen in fingerprints.items():
                        diff = abs(vlen - n_frames_h5)
                        if diff < min_diff:
                            min_diff = diff
                            video_path = vpath
                    if not video_path or min_diff > 60:
                        raise FileNotFoundError(f"指纹匹配失败，找不到接近 {n_frames_h5} 帧的视频！")

                # --- B. 读取 Ground Truth 标签 ---
                positions = h5_data[key]['picks'][()]
                user_summary = h5_data[key]['user_summary'][()]
                cps = h5_data[key]['change_points'][()]
                nfps = h5_data[key]['n_frame_per_seg'][()].tolist()

                # --- C. 推理与评分 (解耦模块) ---
                frame_scores = predict_video_scores(model, video_path, text_tokens, device, config)
                machine_summary = generate_summary(frame_scores, cps, n_frames_h5, nfps, positions)
                
                # 🌟 [关键修正]：严格遵守学术界公平比较铁律，动态切换打分机制！
                eval_metric = 'avg' if dataset_name == 'tvsum' else 'max'
                f1 = evaluate_summary(machine_summary, user_summary, eval_metric=eval_metric)
                
                # 更新计分板
                f1_meter.update(f1)

            except Exception as e:
                print(f"\n🚨 评测视频 {key} 时出错: {e}")

    # 5. 打印最终成绩
    if f1_meter.count > 0:
        print(f"\n🎉=========================================🎉")
        print(f"   [{dataset_name.upper()}] Evaluation Finished!")
        print(f"   🏆 Average F1-score : {f1_meter.avg:.4f}")
        print(f"🎉=========================================🎉")
    else:
        print("\n⚠️ 评估失败，未能成功计算出分数。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="视频摘要评测模块")
    parser.add_argument('--config', required=True, help='配置文件的路径')
    parser.add_argument('--dataset', required=True, choices=['summe', 'tvsum'], help='数据集')
    parser.add_argument('--ckpt', default='exp/ckpt_epoch_49.pth', help='权重路径')
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = CN(yaml.load(f, Loader=yaml.FullLoader))
        
    evaluate(config, args.dataset, args.ckpt)