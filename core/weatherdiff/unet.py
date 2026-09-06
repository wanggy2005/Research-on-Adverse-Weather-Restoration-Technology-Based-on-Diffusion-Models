from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DictConfig:
    """
    把 yaml 字典包装成支持属性访问的对象，
    使移植过来的官方代码里 config.model.ch / config.data.image_size 能直接用。
    """

    def __init__(self, data: Dict[str, Any]):
        for key, value in (data or {}).items():
            setattr(self, str(key), DictConfig(value) if isinstance(value, dict) else value)

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def __repr__(self) -> str:
        return f"DictConfig({self.__dict__})"


# =================================================================== #
# 基础模块（Sinusoidal 时间嵌入、残差块、注意力块、上下采样）
# =================================================================== #

def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """正弦/余弦位置编码，与官方 DDIM 实现一致。"""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class ResnetBlock(nn.Module):
    """带时间嵌入的残差块。"""

    def __init__(self, *, in_ch: int, out_ch: Optional[int] = None,
                 temb_channels: int = 512, dropout: float = 0.0):
        super().__init__()
        out_ch = out_ch or in_ch
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_ch, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_ch)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_ch, eps=1e-6, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        if in_ch != out_ch:
            self.nin_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        h = h + self.temb_proj(F.silu(temb))[:, :, None, None]
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return self.nin_shortcut(x) + h


class AttnBlock(nn.Module):
    """自注意力块（非局部）。"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).view(B, C, H * W).permute(0, 2, 1)
        k = self.k(h).view(B, C, H * W)
        v = self.v(h).view(B, C, H * W)
        attn = torch.bmm(q, k) * (int(C) ** (-0.5))
        attn = F.softmax(attn, dim=2)
        h = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, H, W)
        h = self.proj_out(h)
        return x + h


class Downsample(nn.Module):
    """2× 下采样（卷积实现）。"""

    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Upsample(nn.Module):
    """2× 上采样（最近邻 + 卷积）。"""

    def __init__(self, channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


# =================================================================== #
# DiffusionUNet —— 条件扩散 UNet（ScoreNet 架构）
# =================================================================== #

class DiffusionUNet(nn.Module):
    """
    基于 ScoreNet 的条件扩散 UNet。

    构造签名：DiffusionUNet(config)，其中 config 是 DictConfig 或 dict。
    forward(x, t): x=(B,6,P,P) [条件3通道+噪声3通道], t=(B,) → 返回 (B,3,P,P) 预测噪声
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        ch = config.model.ch
        ch_mult = list(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = list(config.model.attn_resolutions)
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv

        # 输入通道 = 条件(3) + 噪声(3) = 6
        in_channels = config.model.in_channels * 2
        out_ch = config.model.out_ch
        num_resolutions = len(ch_mult)

        # ---- 时间嵌入 MLP (temb.dense) ----
        temb_ch = ch * 4
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(ch, temb_ch),
            nn.Linear(temb_ch, temb_ch),
        ])

        # ---- 输入卷积 ----
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)

        # ---- 编码器 down ----
        # 官方结构：down.i.block 是 ResnetBlock 列表，down.i.attn 是独立的 AttnBlock 列表
        curr_res = config.data.image_size
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        block_in = ch
        for i_level in range(num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(num_res_blocks):
                block.append(ResnetBlock(
                    in_ch=block_in, out_ch=block_out,
                    temb_channels=temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down_block = nn.Module()
            down_block.block = block
            down_block.attn = attn
            if i_level != num_resolutions - 1:
                down_block.downsample = Downsample(block_in, with_conv=resamp_with_conv)
                curr_res //= 2
            self.down.append(down_block)

        # ---- Bottleneck mid（官方用命名属性 block_1 / attn_1 / block_2）----
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_ch=block_in, out_ch=block_in,
            temb_channels=temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_ch=block_in, out_ch=block_in,
            temb_channels=temb_ch, dropout=dropout)

        # ---- 解码器 up ----
        # 官方用 insert(0, ...) 使 self.up[i_level] 恰好对应 i_level
        # 每个 ResnetBlock 都拼接一个 skip，且最后一个 block 的 skip_in 会切换
        self.up = nn.ModuleList()
        for i_level in reversed(range(num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                if i_block == num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(
                    in_ch=block_in + skip_in, out_ch=block_out,
                    temb_channels=temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up_block = nn.Module()
            up_block.block = block
            up_block.attn = attn
            if i_level != 0:
                up_block.upsample = Upsample(block_in, with_conv=resamp_with_conv)
                curr_res *= 2
            self.up.insert(0, up_block)  # prepend 保证索引 = i_level

        # ---- 输出头 ----
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

        self.num_resolutions = num_resolutions
        self.num_res_blocks = num_res_blocks

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 6, P, P)  concat[条件patch(3通道), 噪声patch(3通道)]
        t: (B,) 或 (1,)   时间步
        返回: (B, 3, P, P) 预测噪声
        """
        # 时间嵌入
        temb = get_timestep_embedding(t, self.temb.dense[0].in_features)
        temb = F.silu(self.temb.dense[0](temb))
        temb = self.temb.dense[1](temb)

        # 编码器：每个 ResnetBlock 的输出都入栈，downsample 的输出也入栈
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # Bottleneck
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # 解码器：每个 block 都拼接一个 skip，本级跑完再上采样
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # 输出
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h


def build_unet(config: Dict[str, Any]):
    """
    工厂函数：restorer.py 只调用它，不关心网络细节。

    :param config: configs/*.yaml 加载出来的字典
    :return: torch.nn.Module
    """
    return DiffusionUNet(DictConfig(config))
