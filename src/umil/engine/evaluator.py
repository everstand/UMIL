import os
import torch
import numpy as np
import h5py
import decord
import logging
from tqdm import tqdm
import clip

from umil.metrics.summary_protocol import generate_summary
from umil.metrics.fscore import evaluate_summary
from umil.metrics.diversity import get_summ_diversity
from umil.datasets.metadata.tvsum_metadata import TVSUM_STATIC_MAP

# 暂时保留对旧目录的兼容，等下一步我们再去重构模型层
from models.xclip import build_model 

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count

class VideoEvaluator:
    """
    [Engine Layer] 视频摘要离线评估引擎
    职责：加载权重 -> 抽取视频帧 -> 前向传播 -> 调用协议层算分
    """
    def __init__(self, config, dataset_name, checkpoint_path, candidate_actions):
        self.config = config
        self.dataset_name = dataset_name.lower()
        self.checkpoint_path = checkpoint_path
        self.candidate_actions = candidate_actions
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 严格校验数据集路径
        if self.dataset_name == 'summe':
            self.h5_path = "data/features/eccv16_dataset_summe_google_pool5.h5"
            self.video_dir = "data/raw/SumMe/videos"
        elif self.dataset_name == 'tvsum':
            self.h5_path = "data/features/eccv16_dataset_tvsum_google_pool5.h5"
            self.video_dir = "data/raw/TVSum/videos"
        else:
            raise ValueError(f"🚨 不支持的数据集: {dataset_name}")
            
        # 兼容旧路径(如果你还没把数据移到 data/features，可以暂时保留这几行容错)
        if not os.path.exists(self.h5_path):
            self.h5_path = self.h5_path.replace("data/features/", "data/eccv16_datasets/")
            self.video_dir = self.video_dir.replace("data/raw/", "data/")

    def _predict_video_scores(self, model, video_path, text_tokens):
        """纯粹的单视频前向传播逻辑"""
        vr = decord.VideoReader(video_path, width=224, height=224)
        total_frames = len(vr)
        
        clip_len = self.config.DATA.NUM_FRAMES
        frame_interval = self.config.DATA.FRAME_INTERVAL
        actual_clip_len = clip_len * frame_interval

        frame_scores = []
        for start_idx in range(0, total_frames, actual_clip_len):
            end_idx = min(start_idx + actual_clip_len, total_frames)
            actual_len = end_idx - start_idx
            frame_indices = list(range(start_idx, end_idx, frame_interval))
            
            if len(frame_indices) < clip_len:
                frame_indices.extend([frame_indices[-1]] * (clip_len - len(frame_indices)))
                
            frames = vr.get_batch(frame_indices).asnumpy()
            frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
            frames_tensor = frames_tensor / 255.0
            
            with torch.no_grad():
                outputs = model(frames_tensor, text_tokens)
                
            logits = outputs.get('predicts', outputs.get('logits', outputs.get('output', list(outputs.values())[0]))) if isinstance(outputs, dict) else (outputs[0] if isinstance(outputs, tuple) else outputs)
            score = torch.max(torch.sigmoid(logits), dim=1)[0].item()
            frame_scores.extend([score] * actual_len)

        return np.array(frame_scores)

    def run(self, test_keys):
        """执行评估流水线"""
        if not test_keys or len(test_keys) == 0:
            raise ValueError("🚨 必须提供 test_keys，严禁全量盲测。")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"🚨 找不到权重文件: {self.checkpoint_path}")

        logger.info(f"==> Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model = build_model(state_dict=new_state_dict, T=self.config.DATA.NUM_FRAMES, logger=logger).to(self.device)
        model.eval()

        text_prompts = [f"A video of a person {action}" for action in self.candidate_actions]
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        f1_meter = AverageMeter()
        div_meter = AverageMeter() 

        with h5py.File(self.h5_path, 'r') as h5_data:
            keys = list(h5_data.keys())
            for key in tqdm(keys, desc=f"Eval {self.dataset_name.upper()}", ncols=80, leave=False):
                if self.dataset_name == 'summe':
                    real_name = h5_data[key]['video_name'][()].item().decode('utf-8')
                else:
                    if key not in TVSUM_STATIC_MAP:
                        raise KeyError(f"TVSUM_STATIC_MAP 缺失键值: {key}")
                    real_name = TVSUM_STATIC_MAP[key]

                if real_name not in test_keys:
                    continue
                    
                try:
                    video_path = os.path.join(self.video_dir, f"{real_name}.mp4")
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"找不到视频物理文件: {video_path}")

                    n_frames_h5 = h5_data[key]['n_frames'][()]
                    positions = h5_data[key]['picks'][()]
                    user_summary = h5_data[key]['user_summary'][()]
                    cps = h5_data[key]['change_points'][()]
                    nfps = h5_data[key]['n_frame_per_seg'][()].tolist()
                    seq_features = h5_data[key]['features'][()]

                    # 核心解耦点：调用内部打分器 -> 调用外部协议尺子
                    frame_scores = self._predict_video_scores(model, video_path, text_tokens)
                    machine_summary = generate_summary(frame_scores, cps, n_frames_h5, nfps, positions)

                    eval_metric = 'avg' if self.dataset_name == 'tvsum' else 'max'
                    f1 = evaluate_summary(machine_summary, user_summary, eval_metric=eval_metric)
                    f1_meter.update(f1)

                    # 🌟 核心修正：极其严苛的 Adapter 防御与映射
                    positions = np.asarray(positions).astype(np.int64)
                    # 防御 1：剔除可能存在的越界索引 (H5 预处理的常见坑)
                    positions = positions[positions < len(machine_summary)]

                    # 防御 2：绝不容忍特征与采样点数量不对齐
                    if len(positions) != len(seq_features):
                        raise ValueError(
                            f"🚨 Adapter 层对齐失败：物理抽帧点数量 ({len(positions)}) "
                            f"与视觉特征序列长度 ({len(seq_features)}) 无法完美匹配！"
                        )

                    machine_summary_feature_level = machine_summary[positions]

                    # 严格对齐后，再送入纯净的底层协议
                    diversity = get_summ_diversity(machine_summary_feature_level, seq_features)
                    div_meter.update(diversity)

                except Exception as e:
                    logger.error(f"Error evaluating {key} ({real_name}): {e}")
                    raise e

        if f1_meter.count > 0:
            logger.info(f"[{self.dataset_name.upper()}] Final F-score: {f1_meter.avg:.4f} | Diversity: {div_meter.avg:.4f}")
            return f1_meter.avg, div_meter.avg
        return 0.0, 0.0