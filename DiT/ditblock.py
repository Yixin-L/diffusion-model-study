'''
Author: Yixin Liu yixin_lucy@163.com
Date: 2025-02-21 18:14:12
LastEditors: Yixin Liu yixin_lucy@163.com
LastEditTime: 2025-02-21 20:18:07
FilePath: \undefinedd:\LYX\after_grad\2025年博一下\video generation\ditblock.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) # unusqueeze(1) 相当于增加一个维度

class DitBlock(nn.Module):
    def __init__(self, hidden_size, num_heads,mlp_ratio = 4.0):
        super(DitBlock, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine = False, eps = 1e-6)
        self.attn = Attention(hidden_size, num_heads = num_heads, qkv_bias = True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine= False, eps = 1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda : nn.GELU(approximate = 'tanh')
        self.mlp = Mlp(in_features = hidden_size, hidden_features = mlp_hidden_dim, act_layer = approx_gelu)
        self.adaLN_modulation = nn.Sequential( # MLP生成6个偏移参数
            nn.SiLU(),
            nn.Linear(hidden_size, 6*hidden_size, bias = True)
        )

    def forward(self, x, c):
        # x 来自 input token
        # c condition
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim = 1) # 输出是6*hidden_size, 通过chunk()进行分割
        x = x + self.attn(modulate(self.norm1(x), shift_msa, scale_msa)) * gate_msa.unsqueeze(1)
        x = x + self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) * gate_mlp.unsqueeze(1)
        return x

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super(FinalLayer, self).__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps = 1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias = True)
        self.adaLN_modulation = nn.Sequential( 
            nn.SiLU(),
            nn.Linear(hidden_size, 2*hidden_size, bias = True)
        )
    
    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = self.linear(modulate(self.norm_final(x), shift, scale))
        return x