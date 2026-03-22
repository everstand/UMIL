import os
import time
import shutil
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange

from utils.tools import AverageMeter, epoch_saving
from umil.engine.evaluator import VideoEvaluator

class VideoTrainer:
    """
    [Engine Layer] 弱监督视频摘要训练引擎
    """
    def __init__(self, config, args, model, optimizer, lr_scheduler, train_loader, text_labels, scaler, logger, writer, test_h5_keys):
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
        self.test_h5_keys = test_h5_keys
        self.use_amp = (config.TRAIN.OPT_LEVEL != 'O0')

    def train_one_epoch(self, epoch):
        self.model.train()
        self.optimizer.zero_grad()
        
        num_steps = len(self.train_loader)
        batch_time = AverageMeter()
        tot_loss_meter = AverageMeter()
        mil_loss_meter = AverageMeter()

        end = time.time()
        
        for idx, batch_data in enumerate(self.train_loader):
            torch.cuda.empty_cache()
            images = batch_data["imgs"].cuda(non_blocking=True)
            label_id = batch_data["label"].cuda(non_blocking=True)
            
            bz = images.shape[0]
            n_clips = images.shape[1]
            
            images = rearrange(images, 'b k c t h w -> (b k) t c h w')

            with autocast(enabled=self.use_amp):
                output = self.model(images, self.text_labels)
                
                logits_tc = rearrange(output['y'], '(b k) c -> b k c', b=bz, k=n_clips)
                C = self.text_labels.shape[0]
                labels = label_id.view(bz, C)
                
                K = 3 
                logits_transposed = logits_tc.transpose(1, 2)
                smoothed_logits_transposed = F.avg_pool1d(logits_transposed, kernel_size=K, stride=1, padding=K//2)
                smoothed_logits_tc = smoothed_logits_transposed.transpose(1, 2)
                
                tau = 1.0  
                bag_logits = tau * torch.logsumexp(smoothed_logits_tc / tau, dim=1)
                loss_mil = F.binary_cross_entropy_with_logits(bag_logits, labels.float())
                
                prob_tc = torch.sigmoid(smoothed_logits_tc)  
                sparsity_loss = prob_tc.mean()
                
                total_loss = loss_mil + 0.005 * sparsity_loss
                total_loss = total_loss / self.config.TRAIN.ACCUMULATION_STEPS

            self.scaler.scale(total_loss).backward()
                
            if (idx + 1) % self.config.TRAIN.ACCUMULATION_STEPS == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.lr_scheduler.step_update(epoch * num_steps + idx)

            torch.cuda.synchronize()
            
            tot_loss_meter.update(total_loss.item(), bz)
            mil_loss_meter.update(loss_mil.item(), bz)
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f'Train: [{epoch}/{self.config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                    f'lr {lr:.6f}\t'
                    f'Tot Loss {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                    f'MIL Loss {mil_loss_meter.val:.4f} ({mil_loss_meter.avg:.4f})')
                    
        return tot_loss_meter.avg

    def run(self, start_epoch):
        best_f1_score = 0.0
        
        for epoch in range(start_epoch, self.config.TRAIN.EPOCHS):
            train_loss = self.train_one_epoch(epoch)
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            
            temp_ckpt_path = os.path.join(self.config.OUTPUT, "temp_checkpoint.pth")
            epoch_saving(self.config, epoch, self.model, 0.0, self.optimizer, self.lr_scheduler, self.logger, self.config.OUTPUT, is_best=False)
            
            shutil.move(os.path.join(self.config.OUTPUT, f"ckpt_epoch_{epoch}.pth"), temp_ckpt_path)
            
            evaluator = VideoEvaluator(self.config, self.args.dataset, temp_ckpt_path)
            current_f1, current_div = evaluator.run(test_keys=self.test_h5_keys)
            
            self.writer.add_scalar('Metric/F1_Score', current_f1, epoch)
            
            if current_f1 > best_f1_score:
                best_f1_score = current_f1
                best_ckpt_name = os.path.join(self.config.OUTPUT, f"best_model_split{self.args.split_id}.pth")
                
                shutil.copyfile(temp_ckpt_path, best_ckpt_name)
                self.logger.info(f"New best F1: {best_f1_score:.4f}, saved to {best_ckpt_name}")
                
        self.writer.close()
        self.logger.info(f"Training completed. Best F1: {best_f1_score:.4f}")