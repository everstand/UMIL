import os
import torch
import numpy as np
import h5py
import decord
import logging
from tqdm import tqdm
import clip
import torch.nn.functional as F

from umil.metrics.summary_protocol import generate_summary
from umil.metrics.fscore import evaluate_summary
from umil.metrics.diversity import get_summ_diversity
from umil.datasets.metadata.adapter import build_identity_maps

from models.mil_heads.temporal_smoothing import TemporalSmoothingPrior
from models.mil_heads.representation_score import RepresentationPrior 
from models.builder import build_umil_model

logger = logging.getLogger(__name__)

def preprocess_eval_frames(frames_numpy, input_size, device):
    """
    与 MMCV train_pipeline 严格数学等价的视频帧预处理
    输入: frames_numpy Shape: (T, H, W, C), 值域 [0, 255]
    输出: frames_tensor Shape: (1, T, C, H, W)
    """
    tensors = torch.from_numpy(frames_numpy).permute(0, 3, 1, 2).float().to(device)
    
    scale_resize = int(256 / 224 * input_size)
    _, _, h, w = tensors.shape
    if h < w:
        new_h, new_w = scale_resize, int(w * scale_resize / h)
    else:
        new_h, new_w = int(h * scale_resize / w), scale_resize
        
    tensors = F.interpolate(tensors, size=(new_h, new_w), mode='bilinear', align_corners=False)
    
    top = (new_h - input_size) // 2
    left = (new_w - input_size) // 2
    tensors = tensors[:, :, top:top+input_size, left:left+input_size]
    
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=tensors.dtype, device=tensors.device).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=tensors.dtype, device=tensors.device).view(1, 3, 1, 1)
    tensors = (tensors - mean) / std
    
    return tensors.unsqueeze(0)

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0; self.avg = 0; self.sum = 0; self.count = 0
    def update(self, val, n=1):
        self.val = val; self.sum += val * n; self.count += n
        self.avg = self.sum / self.count

class VideoEvaluator:
    """
    [Engine Layer] 视频摘要离线评估与消融引擎
    """
    def __init__(self, config, dataset_name, checkpoint_path):
        self.config = config
        self.dataset_name = dataset_name.lower()
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        eval_cfg = getattr(config, 'EVAL', None)
        self.ablation_mode = getattr(eval_cfg, 'ABLATION_MODE', 'E1') if eval_cfg else 'E1'
        self.k_classes     = int(getattr(eval_cfg, 'TOP_K', 3)) if eval_cfg else 3
        self.alpha         = float(getattr(eval_cfg, 'ALPHA', 0.5)) if eval_cfg else 0.5
        self.rep_space     = getattr(eval_cfg, 'REP_SPACE', 'raw') if eval_cfg else 'raw'
        
        logger.info(f"Initialize evaluator | Mode: {self.ablation_mode} | K: {self.k_classes} | Alpha: {self.alpha} | R_t Space: {self.rep_space}")

        vocab_path = 'labels/action_vocabulary.txt'
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Missing vocabulary file: {vocab_path}")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.candidate_actions = [line.strip() for line in f if line.strip()]

        self.smoothing_prior = TemporalSmoothingPrior(kernel_size=3).to(self.device)
        self.rep_scorer = RepresentationPrior().to(self.device)
        
        if self.dataset_name == 'summe':
            self.h5_path = "data/eccv16_datasets/eccv16_dataset_summe_google_pool5.h5"
            self.video_dir = "data/SumMe/videos"
        elif self.dataset_name == 'tvsum':
            self.h5_path = "data/eccv16_datasets/eccv16_dataset_tvsum_google_pool5.h5"
            self.video_dir = "data/TVSum/videos"
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.h5_to_real, self.real_to_h5 = build_identity_maps(self.dataset_name, self.h5_path)

    def _select_active_classes(self, video_logits, min_keep=1, max_keep=5, margin=1.5):
        """
        从整段视频的平均 logits 里选出“这个视频真正活跃的语义类”
        不再让任意类别都能在某个窗口里靠 max 抢分。
        """
        video_level_scores = video_logits.mean(dim=0)   # [C]
        max_score = torch.max(video_level_scores)

        keep_mask = video_level_scores >= (max_score - margin)
        keep_indices = torch.nonzero(keep_mask, as_tuple=False).squeeze(1)

        if keep_indices.numel() < min_keep:
            keep_indices = torch.topk(video_level_scores, k=min_keep).indices
        elif keep_indices.numel() > max_keep:
            keep_indices = torch.topk(video_level_scores, k=max_keep).indices

        return keep_indices

    def _predict_video_scores(self, model, video_path, text_tokens):
        vr = decord.VideoReader(video_path)
        total_frames = len(vr)
        
        clip_len = self.config.DATA.NUM_FRAMES
        frame_interval = self.config.DATA.FRAME_INTERVAL
        actual_clip_len = clip_len * frame_interval
        input_size = self.config.DATA.INPUT_SIZE

        all_clip_logits = []
        all_clip_features_raw = []
        all_clip_features_proj = []
        clip_frame_counts = []

        for start_idx in range(0, total_frames, actual_clip_len):
            end_idx = min(start_idx + actual_clip_len, total_frames)
            actual_len = end_idx - start_idx
            clip_frame_counts.append(actual_len)

            frame_indices = list(range(start_idx, end_idx, frame_interval))
            if len(frame_indices) < clip_len:
                frame_indices.extend([frame_indices[-1]] * (clip_len - len(frame_indices)))
                
            frames = vr.get_batch(frame_indices).asnumpy()
            frames_tensor = preprocess_eval_frames(frames, input_size, self.device)
            
            with torch.no_grad():
                outputs = model(frames_tensor, text_tokens)
                
            if not all(k in outputs for k in ('y', 'feature_v_raw', 'feature_v_proj')):
                raise KeyError("Missing required keys in model outputs ('y', 'feature_v_raw', 'feature_v_proj')")
            
            all_clip_logits.append(outputs['y'])
            all_clip_features_raw.append(outputs['feature_v_raw'])
            all_clip_features_proj.append(outputs['feature_v_proj'])

        if not all_clip_logits:
            return np.array([])

        video_logits = torch.cat(all_clip_logits, dim=0)     
        video_features_raw = torch.cat(all_clip_features_raw, dim=0) 
        video_features_proj = torch.cat(all_clip_features_proj, dim=0)

        # 先做时间平滑
        video_logits_3d = video_logits.unsqueeze(0) 
        smoothed_logits = self.smoothing_prior(video_logits_3d).squeeze(0)   # [N, C]
        probs = torch.sigmoid(smoothed_logits)

        # 关键修正：先从整段视频里选“活跃语义类”，再给每个窗口打分
        active_idx = self._select_active_classes(video_logits, min_keep=1, max_keep=5, margin=1.5)
        active_probs = probs[:, active_idx]   # [N, C_active]

        if self.ablation_mode == 'E0':
            # 不平滑对照组：仍然限定在活跃语义子空间里
            raw_probs = torch.sigmoid(video_logits)
            raw_active_probs = raw_probs[:, active_idx]
            final_clip_scores = raw_active_probs.mean(dim=1).cpu().numpy()

        elif self.ablation_mode == 'E1':
            # 默认主模式：活跃类平均，不再对全词表 max
            final_clip_scores = active_probs.mean(dim=1).cpu().numpy()

        elif self.ablation_mode in ['E2', 'E3']:
            k = min(self.k_classes, active_probs.shape[1])
            topk_probs, _ = torch.topk(active_probs, k, dim=1)
            p_scores = topk_probs.mean(dim=1).cpu().numpy()

            if self.ablation_mode == 'E2':
                final_clip_scores = p_scores

            elif self.ablation_mode == 'E3':
                if self.rep_space == 'raw':
                    selected_features = video_features_raw
                elif self.rep_space == 'proj':
                    selected_features = video_features_proj
                else:
                    raise ValueError(f"Unknown R_t space configuration: {self.rep_space}")

                video_features_3d = selected_features.detach().unsqueeze(0)
                r_scores = self.rep_scorer(video_features_3d).squeeze(0).cpu().numpy()
                final_clip_scores = self.alpha * p_scores + (1.0 - self.alpha) * r_scores

        else:
            raise ValueError(f"Unknown ablation mode: {self.ablation_mode}")

        frame_scores = []
        for score, count in zip(final_clip_scores, clip_frame_counts):
            frame_scores.extend([score] * count)

        return np.array(frame_scores)

    def run(self, test_keys):
        """执行评估流水线"""
        if not test_keys or len(test_keys) == 0:
            raise ValueError("test_keys is required.")

        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        logger.info(f"Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model = build_umil_model(self.config, state_dict=new_state_dict, is_training=False, logger=logger).to(self.device)
        model.eval()

        text_prompts = [f"A video of a person {action}" for action in self.candidate_actions]
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        f1_meter = AverageMeter()
        div_meter = AverageMeter() 

        with h5py.File(self.h5_path, 'r') as h5_data:
            keys = list(h5_data.keys())
            for key in tqdm(keys, desc=f"Eval {self.dataset_name.upper()}", ncols=80, leave=False):
                if key not in test_keys:
                    continue
                    
                if key not in self.h5_to_real:
                    raise KeyError(f"Missing H5 key in adapter map: {key}")
                real_name = self.h5_to_real[key]
                    
                try:
                    video_path = os.path.join(self.video_dir, f"{real_name}.mp4")
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"Video file not found: {video_path}")

                    n_frames_h5 = h5_data[key]['n_frames'][()]
                    positions = h5_data[key]['picks'][()]
                    user_summary = h5_data[key]['user_summary'][()]
                    cps = h5_data[key]['change_points'][()]
                    nfps = h5_data[key]['n_frame_per_seg'][()].tolist()
                    seq_features = h5_data[key]['features'][()]

                    frame_scores = self._predict_video_scores(model, video_path, text_tokens)
                    machine_summary = generate_summary(frame_scores, cps, n_frames_h5, nfps, positions)

                    eval_metric = 'avg' if self.dataset_name == 'tvsum' else 'max'
                    f1 = evaluate_summary(machine_summary, user_summary, eval_metric=eval_metric)
                    f1_meter.update(f1)

                    positions = np.asarray(positions).astype(np.int64)
                    positions = positions[positions < len(machine_summary)]

                    if len(positions) != len(seq_features):
                        raise ValueError(
                            f"Adapter mapping mismatch: {len(positions)} physical picks vs {len(seq_features)} visual features"
                        )

                    machine_summary_feature_level = machine_summary[positions]
                    diversity = get_summ_diversity(machine_summary_feature_level, seq_features)
                    div_meter.update(diversity)

                except Exception as e:
                    logger.error(f"Error evaluating {key} ({real_name}): {e}")
                    raise e

        if f1_meter.count > 0:
            logger.info(f"[{self.dataset_name.upper()}] Final F-score: {f1_meter.avg:.4f} | Diversity: {div_meter.avg:.4f}")
            return f1_meter.avg, div_meter.avg
        return 0.0, 0.0