import os
import sys

# 把 UMIL 根目录强行置顶！
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 镇压 decord 的 EOF 崩溃 Bug
os.environ['DECORD_EOF_RETRY_MAX'] = '20480'

# 🌟 2. 雷达扩大后，再安安心心地导入其他包
import logging
import argparse
import h5py
import torch
import numpy as np
import decord
from tqdm import tqdm
import yaml
from yacs.config import CfgNode as CN

# 这次绝对能找到了，因为它就在 UMIL/clip 下
import clip 

# 因为你运行的是 python src/evaluate.py，src 已经在默认路径里，所以直接导入 models 和 helpers
from models.xclip import build_model
from helpers.vsum_helper import generate_summary, evaluate_summary, get_summ_diversity

# 下面保留你的 logging.basicConfig 等代码...

# 配置标准日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

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
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count

# TVSum 静态映射表 (必须手工补全 50 个视频的映射，拒绝动态盲猜)
TVSUM_STATIC_MAP = {
    'video_1': 'AwmHb44_ouw',
    'video_2': '98MoyGZKHXc',
    'video_3': 'J0nA4VgnoCo',
    'video_4': 'gzDbaEs1Rlg',
    'video_5': 'XzYM3PfTM4w',
    'video_6': 'HT5vyqe0Xaw',
    'video_7': 'sTEELN-vY30',
    'video_8': 'vdmoEJ5YbrQ',
    'video_9': 'xwqBXPGE9pQ',
    'video_10': 'akI8YFjEmUw',
    'video_11': 'i3wAGJaaktw',
    'video_12': 'Bhxk-O1Y7Ho',
    'video_13': '0tmA_C6XwfM',
    'video_14': '3eYKfiOEJNs',
    'video_15': 'xxdtq8mxegs',
    'video_16': 'WG0MBPpPC6I',  # 已人工核对 (1帧误差)
    'video_17': 'Hl-__g2gn_A',
    'video_18': 'Yi4Ij2NM7U4',
    'video_19': '37rzWOQsNIw',
    'video_20': 'LRw_obCPUt0',
    'video_21': 'cjibtmSLxQ4',
    'video_22': 'b626MiF1ew4',
    'video_23': 'XkqCExn6_Us',
    'video_24': 'GsAD1KT1xo8',
    'video_25': 'PJrm840pAUI',
    'video_26': '91IHQYk1IQM',
    'video_27': 'RBCABdttQmI',
    'video_28': 'z_6gVvQb2d0',
    'video_29': 'fWutDQy1nnY',
    'video_30': '4wU_LUjG5Ic',
    'video_31': 'VuWGsYPqAX8',
    'video_32': 'JKpqYvAdIsw',
    'video_33': 'xmEERLqJ2kU',
    'video_34': 'byxOvuiIJV0',
    'video_35': '_xMr-HKMfVA',
    'video_36': 'WxtbjNsCQ8A',
    'video_37': 'uGu_10sucQo',
    'video_38': 'EE-bNr36nyA',
    'video_39': 'Se3oxnaPsz0',  # 已人工核对 (1帧误差)
    'video_40': 'oDXZc0tZe04',
    'video_41': 'qqR6AEXwxoQ',
    'video_42': 'EYqVtI9YWJA',
    'video_43': 'eQu1rNs0an0',
    'video_44': 'JgHubY5Vw3Y',
    'video_45': 'iVt07TCkFM0',
    'video_46': 'E11zDS9XGzg',
    'video_47': 'NyBmCxDoHJU',
    'video_48': 'kLxoNp-UchI',
    'video_49': 'jcoYJXDG9sw',
    'video_50': '-esJrBWj2d8',
}

def predict_video_scores(model, video_path, text_tokens, device, config):
    # [保留原版前向传播逻辑]
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
            
        if isinstance(outputs, dict):
            logits = outputs.get('predicts', outputs.get('logits', outputs.get('output', list(outputs.values())[0])))
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        
        score = torch.max(torch.sigmoid(logits), dim=1)[0].item()
        frame_scores.extend([score] * actual_len)

    return np.array(frame_scores)

def evaluate(config, dataset_name, checkpoint_path, test_keys):
    """
    执行严格协议下的模型评估。
    必须传入 test_keys，以确保与训练集的严格物理隔离。
    """
    if not test_keys or len(test_keys) == 0:
        raise ValueError("Fatal Error: Must provide 'test_keys' from official splits. Default full-evaluation is prohibited.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if dataset_name == 'summe':
        h5_path = "data/eccv16_datasets/eccv16_dataset_summe_google_pool5.h5"
        video_dir = "data/SumMe/videos"
    elif dataset_name == 'tvsum':
        h5_path = "data/eccv16_datasets/eccv16_dataset_tvsum_google_pool5.h5"
        video_dir = "data/TVSum/videos"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model = build_model(state_dict=new_state_dict, T=config.DATA.NUM_FRAMES, logger=logger).to(device)
    model.eval()

    text_prompts = [f"A video of a person {action}" for action in CANDIDATE_ACTIONS]
    text_tokens = clip.tokenize(text_prompts).to(device)
    
    f1_meter = AverageMeter()
    div_meter = AverageMeter() 

    with h5py.File(h5_path, 'r') as h5_data:
        keys = list(h5_data.keys())
        
        # 仅在评估循环中使用 tqdm 追踪进度，不打印多余文本
        for key in tqdm(keys, desc=f"Eval {dataset_name.upper()}", ncols=80, leave=False):
            if dataset_name == 'summe':
                real_name = h5_data[key]['video_name'][()].item().decode('utf-8')
            else:
                if key not in TVSUM_STATIC_MAP:
                    raise KeyError(f"Missing mapping for {key} in TVSUM_STATIC_MAP.")
                real_name = TVSUM_STATIC_MAP[key]

            if real_name not in test_keys:
                continue
                
            try:
                ext = '.mp4' 
                video_path = os.path.join(video_dir, f"{real_name}{ext}")
                
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"Video file not found: {video_path}")

                n_frames_h5 = h5_data[key]['n_frames'][()]
                positions = h5_data[key]['picks'][()]
                user_summary = h5_data[key]['user_summary'][()]
                cps = h5_data[key]['change_points'][()]
                nfps = h5_data[key]['n_frame_per_seg'][()].tolist()
                seq_features = h5_data[key]['features'][()]

                frame_scores = predict_video_scores(model, video_path, text_tokens, device, config)
                machine_summary = generate_summary(frame_scores, cps, n_frames_h5, nfps, positions)
                
                eval_metric = 'avg' if dataset_name == 'tvsum' else 'max'
                f1 = evaluate_summary(machine_summary, user_summary, eval_metric=eval_metric)
                f1_meter.update(f1)
                
                diversity = get_summ_diversity(machine_summary, seq_features)
                div_meter.update(diversity)

            except Exception as e:
                logger.error(f"Error evaluating {key} ({real_name}): {e}")

    if f1_meter.count > 0:
        logger.info(f"[{dataset_name.upper()}] F-score: {f1_meter.avg:.4f} | Diversity: {div_meter.avg:.4f}")
        return f1_meter.avg, div_meter.avg
    else:
        logger.warning(f"[{dataset_name.upper()}] Evaluation failed. Returning 0.0")
        return 0.0, 0.0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dataset', required=True, choices=['summe', 'tvsum'])
    parser.add_argument('--ckpt', required=True)
    # 本地单独测试时，由于没有 train.py 传入 test_keys，预留一个传入清单的入口
    parser.add_argument('--test_keys', nargs='+', help='List of test video names', required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = CN(yaml.load(f, Loader=yaml.FullLoader))
        
    evaluate(config, args.dataset, args.ckpt, args.test_keys)