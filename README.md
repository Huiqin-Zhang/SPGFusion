# This is official Pytorch implementation of "SPGFusion: Semantic Prior Guided Infrared and visible image fusion via pretrained vision models"

## ✨News:

[2025-7-4] 我们的论文《[SPGFusion: Semantic Prior Guided Infrared and visible image fusion via pretrained vision models]》被**Information Fusion** 正式接收！[[论文下载](https://www.sciencedirect.com/science/article/pii/S1566253525005068)]

## Recommended Environment
 - [ ] torch  2.2.1
 - [ ] open-clip-torch  2.24.0
 - [ ] torchvision 0.17.1
 - [ ] python 3.9.0

## To Test/Train
1. Downloading the pre-trained checkpoint from [best_model_1.pth](https://drive.google.com/drive/folders/1oPuq89ovz5Ue8DsxeBGbW0ZjI6kTDzGC?usp=drive_link) and putting it in **CHECKPOINT/best_model_1.pth**.
2. Downloading the pre-trained CLIP's checkpoint from [open_clip_pytorch_model.bin](https://drive.google.com/drive/folders/1Bh7YVFYid9z8GT9O2wfxillvO-J2H-4E?usp=drive_link) and putting it in **model/ViT-B-16-laion2B/open_clip_pytorch_model.bin**.

## If this work is helpful to you, please cite it as：
```
@article{zhang2026spgfusion,
  title={SPGFusion: Semantic Prior Guided Infrared and visible image fusion via pretrained vision models},
  author={Zhang, Huiqin and Yao, Shihan and Ma, Jiayi and Jiang, Junjun and Zhang, Yanduo and Zhou, Huabing},
  journal={Information Fusion},
  volume={125},
  pages={103433},
  year={2026},
  publisher={Elsevier}
}
```
