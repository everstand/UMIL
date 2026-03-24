import os
import time
import shutil
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange

from utils.tools import AverageMeter, epoch_saving
from umil.engine.evaluator import VideoEvaluator

class VideoTrainer:
    def __init__(self, config, args, model, optimizer, lr_scheduler, train_loader, text_labels, scaler, logger, writer, val_h5_keys=None, test_h5_keys=None):
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
        self.use_amp = (config.TRAIN.OPT_LEVEL != 'O0')

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
        meter_sparse = AverageMeter()

        end = time.time()
        
        for idx, batch_data in enumerate(self.train_loader):
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
                loss_sparse = prob_tc.mean()
                
                loss_tot_raw = loss_mil + 0.005 * loss_sparse
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
            meter_sparse.update(loss_sparse.item(), bz)
            
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % self.config.PRINT_FREQ == 0:
                lr = self.optimizer.param_groups[0]['lr']
                self.logger.info(
                    f'Train: [{epoch}/{self.config.TRAIN.EPOCHS}][{idx}/{num_steps}] '
                    f'lr: {lr:.6f} '
                    f'Loss: {meter_tot_raw.val:.4f} '
                    f'MIL: {meter_mil.val:.4f} '
                    f'Sparse: {meter_sparse.val:.4f}'
                )
                    
        return meter_tot_raw.avg, meter_mil.avg, meter_sparse.avg

    def run(self, start_epoch):
        best_val_f1 = 0.0
        
        if self.val_h5_keys is None or len(self.val_h5_keys) == 0:
            raise ValueError("Error: val_h5_keys is empty.")

        for epoch in range(start_epoch, self.config.TRAIN.EPOCHS):
            tot_raw, mil, sparse = self.train_one_epoch(epoch)
            
            self.writer.add_scalar('Loss/Total_Raw', tot_raw, epoch)
            self.writer.add_scalar('Loss/MIL', mil, epoch)
            self.writer.add_scalar('Loss/Sparse', sparse, epoch)
            
            temp_ckpt_path = os.path.join(self.config.OUTPUT, "temp_checkpoint.pth")
            epoch_saving(self.config, epoch, self.model, 0.0, self.optimizer, self.lr_scheduler, self.logger, self.config.OUTPUT, is_best=False)
            shutil.move(os.path.join(self.config.OUTPUT, f"ckpt_epoch_{epoch}.pth"), temp_ckpt_path)
            
            evaluator = VideoEvaluator(self.config, self.args.dataset, temp_ckpt_path)
            val_f1, val_div = evaluator.run(test_keys=self.val_h5_keys)
            
            self.writer.add_scalar('Metric_Val/F1_Score', val_f1, epoch)
            self.writer.add_scalar('Metric_Val/Diversity', val_div, epoch)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_ckpt_name = os.path.join(self.config.OUTPUT, f"best_model_split{self.args.split_id}.pth")
                shutil.copyfile(temp_ckpt_path, best_ckpt_name)
                self.logger.info(f"Epoch {epoch} | Best Val F1 updated: {best_val_f1:.4f}")
                
        self.writer.close()
        
        final_ckpt_path = os.path.join(self.config.OUTPUT, f"best_model_split{self.args.split_id}.pth")
        test_evaluator = VideoEvaluator(self.config, self.args.dataset, final_ckpt_path)
        test_f1, test_div = test_evaluator.run(test_keys=self.test_h5_keys)
        self.logger.info(f"Test F1: {test_f1:.4f} | Test Diversity: {test_div:.4f}")