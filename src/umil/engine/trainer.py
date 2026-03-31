import os
import time
import shutil
import math
import torch
import numpy as np
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange

from utils.tools import AverageMeter, epoch_saving
from umil.engine.evaluator import VideoEvaluator


class VideoTrainer:
    """
    弱监督视频摘要训练引擎
    当前版本：
    1. 主任务仍为伪多标签包级监督的 ML-MIL（bag-level BCE）
    2. VLM 显著性仅作为正类时间响应分布的先验正则（loss_sal）
    3. 不再使用 saliency bias 直接修改 bag 聚合
    """

    def __init__(
        self,
        config,
        args,
        model,
        optimizer,
        lr_scheduler,
        train_loader,
        text_labels,
        scaler,
        logger,
        writer,
        val_h5_keys=None,
        test_h5_keys=None,
        vid_to_prior_key=None,
    ):
        self.config = config
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.text_labels = text_labels
        self.scaler = scaler
        self.logger = logger
        self.writer = writer

        self.val_h5_keys = val_h5_keys
        self.test_h5_keys = test_h5_keys
        self.vid_to_prior_key = vid_to_prior_key or {}

        self.use_amp = (config.TRAIN.OPT_LEVEL != 'O0')
        self.last_sal_avg = 0.0

        if args.dataset == "summe":
            prior_path = "labels/summe_saliency_priors.npy"
        else:
            prior_path = "labels/tvsum_saliency_priors.npy"

        if os.path.exists(prior_path):
            self.saliency_priors = np.load(prior_path, allow_pickle=True).item()
        else:
            self.saliency_priors = {}

    def train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()

        num_steps = len(self.train_loader)
        acc_steps = self.config.TRAIN.ACCUMULATION_STEPS
        updates_per_epoch = math.ceil(num_steps / acc_steps)
        update_step = 0

        batch_time = AverageMeter()

        meter_tot_raw = AverageMeter()
        meter_tot_scaled = AverageMeter()
        meter_mil = AverageMeter()
        meter_sal = AverageMeter()
        meter_sparse = AverageMeter()
        meter_tv = AverageMeter()
        meter_peak = AverageMeter()

        end = time.time()

        for idx, batch_data in enumerate(self.train_loader):
            images = batch_data["imgs"].cuda(non_blocking=True)
            label_id = batch_data["label"].cuda(non_blocking=True)

            bz = images.shape[0]
            n_clips = images.shape[1]

            # [B, K, C, T, H, W] -> [(B*K), T, C, H, W]
            images = rearrange(images, 'b k c t h w -> (b k) t c h w')

            with autocast(enabled=self.use_amp):
                output = self.model(images, self.text_labels)

                # [(B*K), C] -> [B, K, C]
                logits_tc = rearrange(output['y'], '(b k) c -> b k c', b=bz, k=n_clips)

                C = self.text_labels.shape[0]
                labels = label_id.view(bz, C).float()

                # 时间平滑
                smooth_k = 3
                logits_transposed = logits_tc.transpose(1, 2)   # [B, C, K]
                smoothed_logits_transposed = F.avg_pool1d(
                    logits_transposed,
                    kernel_size=smooth_k,
                    stride=1,
                    padding=smooth_k // 2
                )
                smoothed_logits_tc = smoothed_logits_transposed.transpose(1, 2)  # [B, K, C]

                B, K, C = smoothed_logits_tc.shape

                # =====================================================
                # 1) 主任务：普通 ML-MIL 聚合（不再用显著性先验直接改聚合）
                # =====================================================
                bag_logits = []
                for b in range(B):
                    m_b = int(labels[b].sum().item())
                    m_b = max(1, min(m_b, K))

                    topm_b, _ = torch.topk(smoothed_logits_tc[b], k=m_b, dim=0)  # [m_b, C]
                    bag_logits.append(topm_b.mean(dim=0))                         # [C]

                bag_logits = torch.stack(bag_logits, dim=0)  # [B, C]
                loss_mil = F.binary_cross_entropy_with_logits(bag_logits, labels.float())

                # =====================================================
                # 2) 显著性正则：约束正类时间响应分布贴近 saliency prior
                # =====================================================
                w_sal = getattr(self.config.TRAIN, "W_SAL", 0.1)
                sal_tau = getattr(self.config.TRAIN, "SAL_TAU", 1.0)

                loss_sal = torch.tensor(
                    0.0,
                    device=smoothed_logits_tc.device,
                    dtype=smoothed_logits_tc.dtype
                )
                sal_count = 0

                for b in range(B):
                    raw_vid = batch_data["vid"][b]
                    if torch.is_tensor(raw_vid):
                        raw_vid = int(raw_vid.item())
                    else:
                        raw_vid = int(raw_vid)

                    prior_key = self.vid_to_prior_key.get(raw_vid, None)
                    prior_np = self.saliency_priors.get(prior_key, None)
                    if prior_np is None:
                        continue

                    prior = torch.as_tensor(
                        prior_np,
                        device=smoothed_logits_tc.device,
                        dtype=smoothed_logits_tc.dtype
                    ).flatten()

                    if prior.numel() != K:
                        prior = F.interpolate(
                            prior.unsqueeze(0).unsqueeze(0),
                            size=K,
                            mode='linear',
                            align_corners=False
                        ).squeeze(0).squeeze(0)

                    prior = torch.clamp(prior, min=1e-6)
                    q = prior / prior.sum()   # [K]

                    pos_cls = torch.nonzero(labels[b] > 0, as_tuple=False).squeeze(1)
                    if pos_cls.numel() == 0:
                        continue

                    for c in pos_cls.tolist():
                        # 模型对正类 c 的时间响应分布
                        p = torch.softmax(smoothed_logits_tc[b, :, c] / sal_tau, dim=0)  # [K]

                        # KL(q || p)
                        loss_sal = loss_sal + torch.sum(q * (torch.log(q) - torch.log(p + 1e-6)))
                        sal_count += 1

                if sal_count > 0:
                    loss_sal = loss_sal / sal_count

                # =====================================================
                # 3) 其余辅助量：先算、先记，但暂不进总损失
                # =====================================================
                prob_tc = torch.sigmoid(smoothed_logits_tc)
                loss_sparse = prob_tc.mean()

                active_prob = prob_tc * labels.unsqueeze(1)
                clip_score = active_prob.sum(dim=-1) / labels.sum(dim=-1, keepdim=True).clamp_min(1.0)
                loss_tv = (clip_score[:, 1:] - clip_score[:, :-1]).abs().mean()

                attn = torch.softmax(smoothed_logits_tc, dim=1)
                pos_attn = attn * labels.unsqueeze(1)
                pos_attn = pos_attn / pos_attn.sum(dim=1, keepdim=True).clamp_min(1e-6)
                loss_peak = (pos_attn.pow(2).sum(dim=1) * labels).sum() / labels.sum().clamp_min(1.0)

                # =====================================================
                # 4) 总损失：当前只用 MIL + Saliency Regularizer
                # =====================================================
                loss_tot_raw = loss_mil + w_sal * loss_sal
                loss_tot_scaled = loss_tot_raw / acc_steps

            self.scaler.scale(loss_tot_scaled).backward()

            should_step = ((idx + 1) % acc_steps == 0) or ((idx + 1) == num_steps)
            if should_step:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                self.lr_scheduler.step_update(epoch * updates_per_epoch + update_step)
                update_step += 1

            torch.cuda.synchronize()

            meter_tot_raw.update(loss_tot_raw.item(), bz)
            meter_tot_scaled.update(loss_tot_scaled.item(), bz)
            meter_mil.update(loss_mil.item(), bz)
            meter_sal.update(loss_sal.item() if sal_count > 0 else 0.0, bz)
            meter_sparse.update(loss_sparse.item(), bz)
            meter_tv.update(loss_tv.item(), bz)
            meter_peak.update(loss_peak.item(), bz)

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f'Train: [{epoch}/{self.config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'lr {lr:.6f}\t'
                    f'Tot(Raw) {meter_tot_raw.val:.4f} ({meter_tot_raw.avg:.4f})\t'
                    f'MIL {meter_mil.val:.4f} ({meter_mil.avg:.4f})\t'
                    f'Sal {meter_sal.val:.4f} ({meter_sal.avg:.4f})\t'
                    f'Sparse {meter_sparse.val:.4f} ({meter_sparse.avg:.4f})\t'
                    f'TV {meter_tv.val:.4f} ({meter_tv.avg:.4f})\t'
                    f'Peak {meter_peak.val:.4f} ({meter_peak.avg:.4f})'
                )

        self.last_sal_avg = meter_sal.avg
        return meter_tot_raw.avg, meter_mil.avg, meter_sparse.avg, meter_tv.avg, meter_peak.avg

    def run(self, start_epoch):
        best_val_f1 = -1.0

        if not self.val_h5_keys:
            raise ValueError("val_h5_keys is required.")

        for epoch in range(start_epoch, self.config.TRAIN.EPOCHS):
            tot_raw, mil, sparse, tv, peak = self.train_one_epoch(epoch)

            self.writer.add_scalar('Loss/Total_Raw', tot_raw, epoch)
            self.writer.add_scalar('Loss/MIL', mil, epoch)
            self.writer.add_scalar('Loss/Sal', self.last_sal_avg, epoch)
            self.writer.add_scalar('Loss/Sparse', sparse, epoch)
            self.writer.add_scalar('Loss/TV', tv, epoch)
            self.writer.add_scalar('Loss/Peak', peak, epoch)

            temp_ckpt_path = os.path.join(self.config.OUTPUT, "temp_checkpoint.pth")
            epoch_saving(
                self.config,
                epoch,
                self.model,
                0.0,
                self.optimizer,
                self.lr_scheduler,
                self.logger,
                self.config.OUTPUT,
                is_best=False
            )
            shutil.move(os.path.join(self.config.OUTPUT, f"ckpt_epoch_{epoch}.pth"), temp_ckpt_path)

            evaluator = VideoEvaluator(self.config, self.args.dataset, temp_ckpt_path)
            val_f1, val_div = evaluator.run(test_keys=self.val_h5_keys)

            self.writer.add_scalar('Metric_Val/F1_Score', val_f1, epoch)
            self.writer.add_scalar('Metric_Val/Diversity', val_div, epoch)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_ckpt_name = os.path.join(
                    self.config.OUTPUT,
                    f"best_model_split{self.args.split_id}.pth"
                )
                shutil.copyfile(temp_ckpt_path, best_ckpt_name)
                self.logger.info(
                    f"New best Val F1: {best_val_f1:.4f}, saved to {best_ckpt_name}"
                )

        self.writer.close()

        if not self.test_h5_keys:
            raise ValueError("test_h5_keys is required for final evaluation.")

        final_ckpt_path = os.path.join(
            self.config.OUTPUT,
            f"best_model_split{self.args.split_id}.pth"
        )
        if not os.path.exists(final_ckpt_path):
            raise FileNotFoundError(f"Best checkpoint not found: {final_ckpt_path}")

        self.logger.info(
            f"Training completed. Best Val F1: {best_val_f1:.4f}. Starting test set evaluation."
        )
        test_evaluator = VideoEvaluator(self.config, self.args.dataset, final_ckpt_path)
        test_f1, test_div = test_evaluator.run(test_keys=self.test_h5_keys)
        self.logger.info(f"Test F1: {test_f1:.4f} | Test Diversity: {test_div:.4f}")