# coding: utf-8
import torch
import argparse
import os
import numpy as np
from pathlib import Path
from model.SPGFusion import SPGFusion
import cv2
from device import device
from TaskFusion_dataset import Fusion_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from model.Dino_Clip import DINOiser
from hydra import compose, initialize

def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """
    
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out

def tensor2img(img, is_norm=True):
    img = img.cpu().float().numpy()
    if img.shape[0] == 1:
        img = np.tile(img, (3, 1, 1))
    if is_norm:
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    img = np.transpose(img, (1, 2, 0))  * 255.0
    return img.astype(np.uint8)

def save_img_single(img, name, is_norm=True):
    img = tensor2img(img, is_norm=True)
    img = Image.fromarray(img)
    img.save(name)

def test_all(path=os.path.join(os.getcwd(), 'OUTPUT', 'Time_test1'), data='assets/data', fusionNet=None, model_ex = None, seg = False, args = None, cfg = None):
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dataset = Fusion_dataset('val', ir_path=os.path.join(data, 'MSRS', 'Test_ir1'),vi_path=os.path.join(data, 'MSRS', 'Test_vi1'))
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 8,
        pin_memory=True,
        drop_last=False)
    
    test_loader.n_iter = len(test_loader)
    
    if model_ex is None:
        model_ex = DINOiser(cfg.model).to(device)
        model_ex.load_dino()
        model_ex.to(device)
    
    if fusionNet is None:
        fusionNet = SPGFusion().to(device)
        fusion_weights = 'CHECKPOINT/best_model_1.pth'
        ckpt = torch.load(fusion_weights, map_location='cpu')
        sd = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        missing, unexpected = fusionNet.load_state_dict(sd, strict=False)
        print('[LOAD] missing:', len(missing))
        print('[LOAD] unexpected:', len(unexpected))
        fusionNet.load_state_dict(sd, strict=True)
        
    fusionNet.eval()
    model_ex.eval()

    test_bar = tqdm(test_loader)
    with torch.no_grad():
        for it, (img_vis, img_ir, name) in enumerate(test_bar):
            img_vis = img_vis.to(device)
            img_ir = img_ir.to(device)

            vi_semantic = model_ex(img_vis)
            ir_semantic = model_ex(img_ir)
            fusion_image = fusionNet(img_vis, img_ir, vi_semantic, ir_semantic)

            fused_img = tensor2numpy(fusion_image)
            img_name = str(name[0])
            save_path = out_dir / img_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_pic(fused_img, str(save_path))

def save_pic(outputpic, save_path):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    cv2.imwrite(save_path, outputpic)
def tensor2numpy(img_tensor):
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training Example')
    parser.add_argument('--config', default = 'clip_dinoiser.yaml', help='config file path')
    args = parser.parse_args()
    
    initialize(config_path="configs", version_base=None)            
    cfg = compose(config_name=args.config)
    test_all(args = args, cfg = cfg)