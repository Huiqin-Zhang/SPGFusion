import torch
import torch.nn as nn
import torchvision.transforms as T
from mmseg.ops import resize
from model.Clip_Text import ClipText
NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

class DinoCLIP(nn.Module):
    def __init__(self, clip_backbone, vit_arch="vit_base", vit_patch_size=16, enc_type_feats="k",
                 gamma=0.2, delta=0.99):
        super(DinoCLIP, self).__init__()
        self.vit_arch = vit_arch
        self.enc_type_feats = enc_type_feats
        self.gamma = gamma
        self.vit_patch_size = vit_patch_size
        self.delta = delta

        # ==== build CLIP backbone =====
        self.clip_backbone = ClipText(clip_backbone)
        for param in self.clip_backbone.parameters():
            param.requires_grad = False

    def load_dino(self):
        from model.FOUND.model import FoundModel, get_vit_encoder
        # ==== build DINO backbone =====
        self.vit_encoder, self.initial_dim, self.hook_features = get_vit_encoder(
            self.vit_arch,
            "dino",
            self.vit_patch_size,
            self.enc_type_feats,
        )
        self.vit_encoder.eval()
        for param in self.vit_encoder.parameters():
            param.requires_grad = False

        self.dino_T = NORMALIZE

    def make_input_divisible(self, x: torch.Tensor) -> torch.Tensor:
        """Pad some pixels to make the input size divisible by the patch size."""
        B, _, H_0, W_0 = x.shape
        pad_w = (self.vit_patch_size - W_0 % self.vit_patch_size) % self.vit_patch_size
        pad_h = (self.vit_patch_size - H_0 % self.vit_patch_size) % self.vit_patch_size
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x
    
    @torch.no_grad()
    def extract_feats(self, type_feats="k"):
        nh = self.vit_encoder.blocks[-1].attn.num_heads
        nb_im, nb_tokens, C_qkv = self.hook_features["qkv"].shape
        qkv = (
            self.hook_features["qkv"]
                .reshape(
                nb_im, nb_tokens, 3, nh, C_qkv // nh // 3
            )
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if type_feats == "q":
            return q.transpose(1, 2).float()
        elif type_feats == "k":
            return k.transpose(1, 2).float()
        elif type_feats == "v":
            return v.transpose(1, 2).float()
        else:
            raise ValueError("Unknown features")
    
    @torch.no_grad()
    def get_dino_corrs(self, x: torch.Tensor):
        B = x.shape[0]
        feats, (hf, wf) = self.get_dino_features(x)
        corrs = torch.matmul(feats.permute(0, 2, 1), feats).reshape(B, hf, wf, hf * wf)
        
        if self.gamma is not None:
            corrs[corrs < self.gamma] = 0.0
        return corrs.permute(0, 3, 1, 2)  # B C h w

    def get_dino_features(self, x: torch.Tensor):
        x = self.make_input_divisible(x)
        batch = self.dino_T(x)
        h_featmap = batch.shape[-2] // self.vit_patch_size
        w_featmap = batch.shape[-1] // self.vit_patch_size

        _ = self.vit_encoder(batch)

        # Get decoder features
        feats = self.extract_feats(type_feats=self.enc_type_feats)
        num_extra_tokens = 1

        feats = feats[:, num_extra_tokens:, :, :].flatten(-2, -1).permute(0, 2, 1)  # B C nbtokens
        feats = feats / feats.norm(dim=1, keepdim=True)  # normalize features

        return feats, (h_featmap, w_featmap)


    @torch.no_grad()
    def get_clip_features(self, x: torch.Tensor):
        x = self.make_input_divisible(x)
        feat = self.clip_backbone(x)
        return feat
        
    @staticmethod
    def compute_weighted_pool(maskclip_feats: torch.Tensor, corrs: torch.Tensor):
        B = maskclip_feats.shape[0]
        h_m, w_m = maskclip_feats.shape[-2:]
        h_w, w_w = corrs.shape[-2:]

        if (h_m != h_w) or (w_m != w_w):
            maskclip_feats = resize(
                input=maskclip_feats,
                size=(h_w, w_w),
                mode='bilinear',
                align_corners=False)
            h_m, w_m = h_w, w_w
        maskclip_feats_ref = torch.einsum("bnij, bcij -> bcn", corrs, maskclip_feats)  # B C HW
        norm_factor = corrs.flatten(-2, -1).sum(dim=-1)[:, None]  # B 1 HW
        maskclip_feats_ref = maskclip_feats_ref / (norm_factor + 1e-6)

        maskclip_feats_ref = maskclip_feats_ref.reshape(B, -1, h_m, w_m)
        return maskclip_feats_ref
        
class DINOiser(DinoCLIP):
    def __init__(self, clip_backbone, vit_arch="vit_base", vit_patch_size=16, enc_type_feats="v",
                 feats_idx=-3, gamma=0.2, delta=0.99, in_dim=256, conv_kernel=3):
        super(DINOiser, self).__init__(clip_backbone, vit_arch, vit_patch_size, enc_type_feats, gamma)

        in_size = 768 if feats_idx != 'final' else 512
        self.gamma = gamma
        self.feats_idx = feats_idx
        self.delta = delta
        self.in_dim = in_dim

        if feats_idx != 'final':
            train_feats = {}

            def get_activation(name):
                def hook(model, input, output):
                    train_feats[name] = output.detach().permute(1, 0, 2)  # change to batch first

                return hook
            
            self.clip_backbone.backbone.visual.transformer.resblocks[feats_idx].ln_2.register_forward_hook(
                get_activation('clip_inter'))
            self.train_feats = train_feats

    def forward_pass(self, x: torch.Tensor):
        x = self.make_input_divisible(x)
        clip_proj_feats = self.get_clip_features(x)

        B, c_dim, h, w = clip_proj_feats.shape
        
        if self.feats_idx != 'final':
            clip_feats = self.train_feats['clip_inter']
            B, N, c_dim = clip_feats.shape
            clip_feats = clip_feats[:, 1:, ].permute(0, 2, 1).reshape(B, c_dim, h, w)
        else:
            clip_feats = clip_proj_feats

        return clip_feats

    def forward(self, x: torch.Tensor):
        output = self.forward_pass(x)
        dino_corrs = self.get_dino_corrs(x)
        out_feats = self.compute_weighted_pool(output, dino_corrs)
        return out_feats