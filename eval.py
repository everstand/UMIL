import os
import argparse
import h5py
import torch
import numpy as np
import decord
from tqdm import tqdm
import yaml
from yacs.config import CfgNode as CN
import torch.nn.functional as F
import clip  # <--- 引入 CLIP 用于文本处理

# 导入你的模型和评测工具
from models.xclip import build_model
from vsum_tools import generate_summary, evaluate_summary

# =========== 补充的全局词表 (UMIL 必须的 text prompt) ===========
candidate_actions = [
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
text_prompts = [f"A video of a person {action}" for action in candidate_actions]
# ===============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='视频摘要统一评测脚本')
    parser.add_argument('--config', required=True, help='配置文件的路径')
    parser.add_argument('--dataset', required=True, choices=['summe', 'tvsum'], help='要评测的数据集 (summe 或 tvsum)')
    parser.add_argument('--ckpt', default='exp/ckpt_epoch_49.pth', help='模型权重的路径')
    return parser.parse_args()

def evaluate_model(config, dataset_name, checkpoint_path):
    print(f"🚀 开始执行 CVPR 级视频摘要官方评测 | 当前数据集: {dataset_name.upper()}")

    # ================= 动态路由 =================
    if dataset_name == 'summe':
        H5_FILE_PATH = "data/eccv16_datasets/eccv16_dataset_summe_google_pool5.h5"
        VIDEO_DIR = "data/SumMe/videos"
    elif dataset_name == 'tvsum':
        H5_FILE_PATH = "data/eccv16_datasets/eccv16_dataset_tvsum_google_pool5.h5"
        VIDEO_DIR = "data/TVSum/videos"
    else:
        raise ValueError(f"🚨 未知的数据集: {dataset_name}")

    # ================= 1. 加载模型与权重 =================
    print(f"📦 正在加载权重: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"🚨 找不到权重文件: {checkpoint_path}，请先训练！")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    model = build_model(state_dict=new_state_dict, T=config.DATA.NUM_FRAMES, logger=logger)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    # ================= 🚀 处理文本 Token (UMIL 核心要求) =================
    print("🔤 正在编码 45 维文本标签字典...")
    text_tokens = clip.tokenize(text_prompts).to(device)
    # =========================================================================

    # ================= 🚀 自动构建“帧数指纹”匹配库 =================
    print(f"🔍 正在扫描 {VIDEO_DIR} 中的视频真实帧数，构建匹配指纹库...")
    video_fingerprints = {}
    if os.path.exists(VIDEO_DIR):
        for fname in os.listdir(VIDEO_DIR):
            if fname.endswith(('.mp4', '.avi', '.webm', '.mkv')):
                vpath = os.path.join(VIDEO_DIR, fname)
                try:
                    vr_tmp = decord.VideoReader(vpath)
                    video_fingerprints[vpath] = len(vr_tmp)
                except Exception:
                    pass
    print(f"✅ 指纹库构建完毕，共探测到 {len(video_fingerprints)} 个视频文件。")
    # =========================================================================

    all_f1_scores = []

    print(f"📂 正在加载官方 H5 标签库: {H5_FILE_PATH}")
    with h5py.File(H5_FILE_PATH, 'r') as h5_data:
        vid_keys = list(h5_data.keys())
        
        for vid_key in tqdm(vid_keys, desc="评测进度", ncols=100):
            video_name = str(vid_key) # 默认兜底名字
            try:
                n_frames_h5 = h5_data[vid_key]['n_frames'][()]
                
                # ================= 动态名字提取 & 自动指纹匹配 =================
                vid_group_keys = list(h5_data[vid_key].keys())
                video_path = None
                
                if 'video_name' in vid_group_keys:
                    vid_name_obj = h5_data[vid_key]['video_name'][()]
                    if hasattr(vid_name_obj, 'item'): vid_name_obj = vid_name_obj.item()
                    video_name = vid_name_obj.decode('utf-8') if isinstance(vid_name_obj, bytes) else str(vid_name_obj)
                    
                    for ext in ['.mp4', '.avi', '.webm', '.mkv']:
                        tmp_path = os.path.join(VIDEO_DIR, f"{video_name}{ext}")
                        if os.path.exists(tmp_path):
                            video_path = tmp_path
                            break

                if video_path is None or not os.path.exists(video_path):
                    min_diff = float('inf')
                    for vpath, vlen in video_fingerprints.items():
                        diff = abs(vlen - n_frames_h5)
                        if diff < min_diff:
                            min_diff = diff
                            video_path = vpath
                    
                    if video_path is None or min_diff > 60:
                        raise FileNotFoundError(f"指纹匹配失败: 找不到与 {vid_key} (要求帧数: {n_frames_h5}) 相近的视频！")
                        
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                # ==========================================================

                n_frames = n_frames_h5
                positions = h5_data[vid_key]['picks'][()]
                user_summary = h5_data[vid_key]['user_summary'][()]
                cps = h5_data[vid_key]['change_points'][()]
                nfps = h5_data[vid_key]['n_frame_per_seg'][()].tolist()

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
                        pad_len = clip_len - len(frame_indices)
                        frame_indices.extend([frame_indices[-1]] * pad_len)
                        
                    frames = vr.get_batch(frame_indices).asnumpy()
                    
                    frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0).to(device)
                    frames_tensor = frames_tensor / 255.0
                    
                    with torch.no_grad():
                        # 【核心修复】：把 text_tokens 一并传给模型！
                        outputs = model(frames_tensor, text_tokens)
                        
                    # === 智能拆开模型的输出字典/元组 ===
                    if isinstance(outputs, dict):
                        # 尝试抓取大模型最常用的预测结果键名
                        if 'predicts' in outputs:
                            logits = outputs['predicts']
                        elif 'logits' in outputs:
                            logits = outputs['logits']
                        elif 'output' in outputs:
                            logits = outputs['output']
                        else:
                            # 如果都不叫，强行拿字典里的第一个张量
                            logits = list(outputs.values())[0]
                    elif isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # 确保拿到的一定是 Tensor
                    score = torch.max(torch.sigmoid(logits), dim=1)[0].item()
                    # ===================================
                    frame_scores.extend([score] * actual_len)

                frame_scores = np.array(frame_scores)
                machine_summary = generate_summary(frame_scores, cps, n_frames, nfps, positions)
                f1 = evaluate_summary(machine_summary, user_summary, eval_metric='avg')
                all_f1_scores.append(f1)

            except Exception as e:
                print(f"\n🚨 评测 {vid_key} ({video_name}) 时出错: {e}")

    if all_f1_scores:
        mean_f1 = np.mean(all_f1_scores)
        print(f"\n🎉=========================================🎉")
        print(f"   评测圆满结束！在 {dataset_name.upper()} 数据集上的成绩：")
        print(f"   🏆 Average F1-score : {mean_f1:.4f}")
        print(f"🎉=========================================🎉")
    else:
        print("\n⚠️ 没能成功算出 F1-score，请检查上面的报错信息。")

if __name__ == '__main__':
    args = parse_args()
    
    with open(args.config, "r") as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    config = CN(config_dict)
    
    evaluate_model(config, dataset_name=args.dataset, checkpoint_path=args.ckpt)