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
from umil.datasets.metadata.adapter import build_identity_maps

from models.mil_heads.temporal_smoothing import TemporalSmoothingPrior
from models.mil_heads.representation_score import RepresentationPrior 
from models.builder import build_umil_model

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
    [Engine Layer] 视频摘要离线评估与消融引擎
    """
    def __init__(self, config, dataset_name, checkpoint_path):
        self.config = config
        self.dataset_name = dataset_name.lower()
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. 防御性配置读取与双流形参数解析
        eval_cfg = getattr(config, 'EVAL', None)
        self.ablation_mode = getattr(eval_cfg, 'ABLATION_MODE', 'E3') if eval_cfg else 'E3'
        self.k_classes     = int(getattr(eval_cfg, 'TOP_K', 3)) if eval_cfg else 3
        self.alpha         = float(getattr(eval_cfg, 'ALPHA', 0.5)) if eval_cfg else 0.5
        self.rep_space     = getattr(eval_cfg, 'REP_SPACE', 'raw') if eval_cfg else 'raw'
        
        logger.info(f"🚀 初始化评估引擎 | 模式: {self.ablation_mode} | K: {self.k_classes} | Alpha: {self.alpha} | R_t空间: {self.rep_space}")

        # 2. 强制语义同源：从唯一源读取动作词表
        vocab_path = 'labels/action_vocabulary.txt'
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"🚨 找不到统一动作词表文件: {vocab_path}，请确保伪标签生成逻辑已先运行。")
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.candidate_actions = [line.strip() for line in f if line.strip()]

        # 3. 实例化组件
        self.smoothing_prior = TemporalSmoothingPrior(kernel_size=3).to(self.device)
        self.rep_scorer = RepresentationPrior().to(self.device)
        
        # 4. 数据集路径校验
        if self.dataset_name == 'summe':
            self.h5_path = "data/eccv16_datasets/eccv16_dataset_summe_google_pool5.h5"
            self.video_dir = "data/SumMe/videos"
        elif self.dataset_name == 'tvsum':
            self.h5_path = "data/eccv16_datasets/eccv16_dataset_tvsum_google_pool5.h5"
            self.video_dir = "data/TVSum/videos"
        else:
            raise ValueError(f"🚨 不支持的数据集: {dataset_name}")

        self.h5_to_real, self.real_to_h5 = build_identity_maps(self.dataset_name, self.h5_path)


    def _predict_video_scores(self, model, video_path, text_tokens):
        vr = decord.VideoReader(video_path, width=224, height=224)
        total_frames = len(vr)
        
        clip_len = self.config.DATA.NUM_FRAMES
        frame_interval = self.config.DATA.FRAME_INTERVAL
        actual_clip_len = clip_len * frame_interval

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
            frames_tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
            frames_tensor = frames_tensor / 255.0
            
            with torch.no_grad():
                outputs = model(frames_tensor, text_tokens)
                
            if not all(k in outputs for k in ('y', 'feature_v_raw', 'feature_v_proj')):
                raise KeyError("模型输出缺失主键 'y', 'feature_v_raw' 或 'feature_v_proj'")
            
            all_clip_logits.append(outputs['y'])
            all_clip_features_raw.append(outputs['feature_v_raw'])
            all_clip_features_proj.append(outputs['feature_v_proj'])

        if not all_clip_logits:
            return np.array([])

        video_logits = torch.cat(all_clip_logits, dim=0)     
        video_features_raw = torch.cat(all_clip_features_raw, dim=0) 
        video_features_proj = torch.cat(all_clip_features_proj, dim=0)

        if self.ablation_mode == 'E0':
            probs = torch.sigmoid(video_logits)
            p_scores = torch.max(probs, dim=1)[0].cpu().numpy()
            final_clip_scores = p_scores
        else:
            video_logits_3d = video_logits.unsqueeze(0) 
            smoothed_logits = self.smoothing_prior(video_logits_3d).squeeze(0)
            probs = torch.sigmoid(smoothed_logits)

            if self.ablation_mode == 'E1':
                p_scores = torch.max(probs, dim=1)[0].cpu().numpy()
                final_clip_scores = p_scores
            elif self.ablation_mode in ['E2', 'E3']:
                k = min(self.k_classes, probs.shape[1])
                topk_probs, _ = torch.topk(probs, k, dim=1)
                p_scores = topk_probs.mean(dim=1).cpu().numpy()

                if self.ablation_mode == 'E2':
                    final_clip_scores = p_scores
                elif self.ablation_mode == 'E3':
                    # 依据配置进行流形空间路由
                    if self.rep_space == 'raw':
                        selected_features = video_features_raw
                    elif self.rep_space == 'proj':
                        selected_features = video_features_proj
                    else:
                        raise ValueError(f"未知的 R_t 空间配置: {self.rep_space}")

                    video_features_3d = selected_features.detach().unsqueeze(0)
                    r_scores = self.rep_scorer(video_features_3d).squeeze(0).cpu().numpy()
                    
                    # 终极融合: S_t = αP_t + (1-α)R_t
                    final_clip_scores = self.alpha * p_scores + (1.0 - self.alpha) * r_scores
                else:
                    raise ValueError(f"未知的消融模式: {self.ablation_mode}")

        frame_scores = []
        for score, count in zip(final_clip_scores, clip_frame_counts):
            frame_scores.extend([score] * count)

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

        model = build_umil_model(self.config, state_dict=new_state_dict, is_training=False, logger=logger).to(self.device)
        model.eval()

        text_prompts = [f"A video of a person {action}" for action in self.candidate_actions]
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        f1_meter = AverageMeter()
        div_meter = AverageMeter() 

        with h5py.File(self.h5_path, 'r') as h5_data:
            keys = list(h5_data.keys())
            for key in tqdm(keys, desc=f"Eval {self.dataset_name.upper()}", ncols=80, leave=False):
                # 直接用 H5 的内部键进行身份拦截
                if key not in test_keys:
                    continue
                    
                # 纯粹的 Adapter 翻译：不再有 if dataset_name == ...
                if key not in self.h5_to_real:
                    raise KeyError(f"身份适配字典缺失 H5 键值: {key}")
                real_name = self.h5_to_real[key]
                    
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

                    # 前向推理
                    frame_scores = self._predict_video_scores(model, video_path, text_tokens)
                    machine_summary = generate_summary(frame_scores, cps, n_frames_h5, nfps, positions)

                    # 评测 F1
                    eval_metric = 'avg' if self.dataset_name == 'tvsum' else 'max'
                    f1 = evaluate_summary(machine_summary, user_summary, eval_metric=eval_metric)
                    f1_meter.update(f1)

                    # 严格对齐 Adapter 层
                    positions = np.asarray(positions).astype(np.int64)
                    positions = positions[positions < len(machine_summary)]

                    if len(positions) != len(seq_features):
                        raise ValueError(
                            f"🚨 Adapter层未对齐: 物理抽帧点数量({len(positions)}) vs 视觉特征序列长度({len(seq_features)})"
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