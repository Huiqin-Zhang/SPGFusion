import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.ops import resize
from typing import List, Tuple
from torch import Tensor
from open_clip import get_tokenizer, create_model_from_pretrained
import torchvision.transforms as T
OPENAI_NORMALIZE = T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

class ClipText(nn.Module):
    def __init__(
            self,
            backbone
        ):
        super(ClipText, self).__init__()
        self.patch_size = backbone.get('patch_size')
        self.img_size = tuple([backbone.get('img_size', 224)]*2)
        pretrained = backbone.get('clip_model_pth')
        model, preprocess = create_model_from_pretrained(backbone.get('clip_model'), pretrained=pretrained)
        model.eval()
        self.clip_T = OPENAI_NORMALIZE
        self.hook_features = {}
        self.backbone = model
        def hook_fn_forward(module, input, output):
            self.hook_features["v"] = output

        self.backbone.visual.transformer.resblocks[-2].register_forward_hook(hook_fn_forward)
        self._positional_embd = nn.Parameter(self.backbone.visual.positional_embedding.data.clone())

    @torch.no_grad()
    def extract_feat(self, inputs: Tensor) -> Tuple[Tensor]:
        """Extract features from images."""
        pos_embed = self.backbone.visual.positional_embedding

        B, C, H, W = inputs.shape
        hw_shape = (H // self.patch_size, W // self.patch_size)
        x_len, pos_len = hw_shape[0]*hw_shape[1], pos_embed.shape[0]

        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    '{}, {}'.format(x_len, pos_len))

            self.backbone.visual.positional_embedding.data = self.resize_pos_embed(
                self._positional_embd[None], hw_shape,  (pos_h, pos_w), 'bicubic')[0]
            
        _ = self.backbone(inputs)
        v = self.hook_features["v"]
        v = self.extract_v(v, self.backbone.visual.transformer.resblocks[-1]).permute(1, 0, 2)
        v = self.backbone.visual.ln_post(v)
        v = v[:, 1:]
        v = v.reshape(B, hw_shape[0], hw_shape[1], -1).permute(0, 3, 1, 2).contiguous()
        self.backbone.visual.positional_embedding.data = self._positional_embd
        return v

    def extract_v(self, x, block):
        y = block.ln_1(x)
        y = torch.nn.functional.linear(y, block.attn.in_proj_weight, block.attn.in_proj_bias)
        B, N, C = y.shape
        y = y.view(B, N, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * B, N, C // 3)
        y = F.linear(y, block.attn.out_proj.weight, block.attn.out_proj.bias)
        q, k, v = y.tensor_split(3, dim=0)
        v += x
        v += block.mlp(block.ln_2(v))
        return v


    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs: Tensor, return_feat=False) -> Tensor:
        inputs = self.clip_T(inputs)
        x = self.extract_feat(inputs)
        return x