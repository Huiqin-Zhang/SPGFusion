# coding: utf-8
import os
import sys
import torch
from time import time
from tensorboardX import SummaryWriter
from datetime import datetime, timedelta
import torch.optim as optim
from tqdm import tqdm
from util import (
    fusion_loss,
    create_lr_scheduler)
from model.SPGFusion import SPGFusion
from device import device
from TaskFusion_dataset import Fusion_dataset
from torch.utils.data import DataLoader
from model.Dino_Clip import DINOiser
import torch.backends.cudnn as cudnn

class SPGTrain:
    def __init__(self, args, cfg):
        self.args = args
        self.fusionNet = SPGFusion().to(device)
        
        self.model_ex = DINOiser(cfg.model).to(device)
        self.model_ex.load_dino()
        self.model_ex.to(device)
        self.model_ex.eval()
        self.loss = fusion_loss()
        cudnn.benchmark = True
        
    def train_step(self, optimizer, lr_scheduler, image_vi, image_ir):
        self.fusionNet.train()
        l_g_total = 0
        
        vi_semantic = self.model_ex(image_vi)
        ir_semantic = self.model_ex(image_ir)
        optimizer.zero_grad()
        fusion_image = self.fusionNet(image_vi, image_ir, vi_semantic, ir_semantic)

        total_loss, loss_ssim, loss_Grad, loss_int, loss_color, loss_consist = self.loss(image_vi, image_ir, fusion_image)

        l_g_total += total_loss
        l_g_total.backward()
        lr_fusion = optimizer.param_groups[0]["lr"]
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        return total_loss, loss_ssim, loss_Grad, loss_int, loss_color, loss_consist, lr_fusion
    
    def train(self):
        train_dataset = Fusion_dataset('train')
        print("the training dataset is length:{}".format(len(train_dataset)))
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True)
        ep_iter = len(train_loader)
        max_iter = self.args.epoch * ep_iter
        print("Training...iter: {}".format(max_iter))
        optimizer = optim.AdamW(self.fusionNet.parameters(), lr=0.0001, weight_decay=5E-2)
        lr_scheduler = create_lr_scheduler(optimizer, ep_iter, self.args.epoch, warmup=True)

        current_time = datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
        summary_dir = os.path.join(self.args.summary_dir, 'Train_Fusion' + current_time)
        with SummaryWriter(summary_dir) as writer:    
            min_l_g_total = float('inf')  
            Allepochs = self.args.epoch
            global_step = 0
            start_epoch = 1
            checkpoint_dir = self.args.savePTH
            if os.path.exists(checkpoint_dir):
                files = os.listdir(checkpoint_dir)
                epochs = [int(f[5:]) for f in files if f.startswith('epoch') and f[5:].isdigit()]
                if epochs:
                    model_save_path = os.path.join(checkpoint_dir, 'epoch{}'.format(max(epochs)), 'best_model.pth')
                    checkpoint = torch.load(model_save_path, map_location='cpu')
                    
                    self.fusionNet.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    start_epoch = checkpoint['ep'] + 1
                    global_step = checkpoint['total_it']
                    min_l_g_total = checkpoint['min_l_g_total']
                    
                    lr = optimizer.param_groups[0]['lr']
                    print(f'lr_fusion= {lr:.10f}')
                    print('Resuming training from epoch {}'.format(start_epoch))
            
            start = glob_st = time()
            for epoch in range(start_epoch, Allepochs + 1):
                data_loader = tqdm(train_loader, file=sys.stdout)
                epoch_loss_sum = 0.0
                epoch_iter_cnt = 0

                for it, (image_vis, image_ir, names) in enumerate(data_loader):
                    global_step += 1
                    batch_images_vi = image_vis.to(device)
                    batch_images_ir = image_ir.to(device)

                    total_loss, loss_ssim, loss_Grad, loss_int, loss_color, loss_consist, lr_fusion = \
                        self.train_step(optimizer, lr_scheduler, batch_images_vi, batch_images_ir)

                    tl = float(total_loss.item() if hasattr(total_loss, "item") else total_loss)
                    epoch_loss_sum += tl
                    epoch_iter_cnt += 1

                    data_loader.desc = (
                        "[train epoch {}] loss: {:.3f}  ssim loss: {:.3f}  grad loss: {:.3f}  "
                        "int loss: {:.3f}  color loss: {:.3f}  consist loss: {:.3f} lr: {:.6f}"
                    ).format(epoch, total_loss, loss_ssim, loss_Grad, loss_int, loss_color, loss_consist, lr_fusion)

                end = time()
                training_time, glob_t_intv = end - start, end - glob_st
                eta = int((Allepochs * len(train_loader) - global_step) * (glob_t_intv / max(global_step, 1)))
                eta = str(timedelta(seconds=eta))
                print(f'Still need {eta}')
                start = time()

                epoch_loss_avg = epoch_loss_sum / max(epoch_iter_cnt, 1)

                improved = epoch_loss_avg < min_l_g_total
                if improved:
                    min_l_g_total = epoch_loss_avg

                print(f'[epoch {epoch}] avg_total_loss={epoch_loss_avg:.6f} | best(min_l_g_total)={min_l_g_total:.6f} | improved={improved}')

                if epoch > 90:
                    if improved or epoch == Allepochs:
                        print('Saving model (improved min_l_g_total or last epoch after 90)...')
                        epoch_folder = os.path.join(self.args.savePTH, f'epoch{epoch}')
                        os.makedirs(epoch_folder, exist_ok=True)
                        save_file = {
                            "model": self.fusionNet.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "ep": epoch,
                            "total_it": global_step,
                            "min_l_g_total": min_l_g_total
                        }
                        torch.save(save_file, os.path.join(epoch_folder, "best_model.pth"))
                        print(f'Epoch {epoch}/{Allepochs}, Model saved at {epoch_folder}, min_l_g_total: {min_l_g_total:.6f}')
                    else:
                        print(f'Model not saved (epoch>{90}). epoch_avg_loss: {epoch_loss_avg:.6f}, best: {min_l_g_total:.6f}')
                else:
                    print(f'Skip saving before or at epoch 90. (epoch={epoch})')
                    if epoch == Allepochs and Allepochs <= 90:
                        print('Final epoch <= 90, saving anyway...')
                        epoch_folder = os.path.join(self.args.savePTH, f'epoch{epoch}')
                        os.makedirs(epoch_folder, exist_ok=True)
                        save_file = {
                            "model": self.fusionNet.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                            "ep": epoch,
                            "total_it": global_step,
                            "min_l_g_total": min_l_g_total
                        }
                        torch.save(save_file, os.path.join(epoch_folder, "best_model.pth"))
                        print(f'Epoch {epoch}/{Allepochs}, Model saved at {epoch_folder}, min_l_g_total: {min_l_g_total:.6f}')