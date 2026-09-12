"""
=============================================================================
Patch-based DDIM 采样算法
=============================================================================

对应官方文件: WeatherDiffusion/utils/sampling.py

本文件实现了扩散模型推理的核心采样逻辑：beta 调度、alpha 计算、
滑窗坐标生成、数据归一化，以及关键的 generalized_steps_overlapping() 函数。

核心思路（论文的关键做法）:
  1. 把整图切成 patch_size × patch_size、步长 grid_r 的重叠小块
  2. 每个去噪步里，对所有 patch 分别预测噪声，按位置累加到整图 buffer
  3. 用"每个像素被多少 patch 覆盖"的 mask 做平均，得到整图噪声估计
  4. 再走一步 DDIM 更新，如此循环 sampling_timesteps 次
这样显存只与 patch 大小和一次并行的 patch 数量有关，与整图分辨率无关。
=============================================================================
"""

from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

# --------------------------------------------------------------------------- #
# 通用工具函数
# --------------------------------------------------------------------------- #
def get_beta_schedule(
    beta_schedule: str = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    num_diffusion_timesteps: int = 1000,
) -> np.ndarray:
    """生成 beta 序列，与官方 DDIM 实现一致。"""
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        betas = (
            np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(f"不支持的 beta_schedule: {beta_schedule}")
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def make_timestep_seq(num_diffusion_timesteps: int, sampling_timesteps: int) -> List[int]:
    """把 1000 步压缩成 sampling_timesteps 步的等间隔序列（DDIM 加速采样）。"""
    sampling_timesteps = max(1, min(sampling_timesteps, num_diffusion_timesteps))
    skip = num_diffusion_timesteps // sampling_timesteps
    return list(range(0, num_diffusion_timesteps, skip))


def overlapping_grid_indices(
    height: int, width: int, output_size: int, r: Optional[int] = None
) -> Tuple[List[int], List[int]]:
    """
    生成滑窗左上角坐标列表。与官方实现保持一致：
    末尾不足一个 patch 的部分靠 preprocess 阶段把尺寸对齐到 16 的倍数来兜住。
    """
    r = 16 if r is None else int(r)
    h_list = list(range(0, max(1, height - output_size + 1), r))
    w_list = list(range(0, max(1, width - output_size + 1), r))
    # 保证右下边界一定被覆盖
    if h_list[-1] != height - output_size and height - output_size > 0:
        h_list.append(height - output_size)
    if w_list[-1] != width - output_size and width - output_size > 0:
        w_list.append(width - output_size)
    return h_list, w_list


def corner_list(height: int, width: int, patch_size: int, grid_r: int) -> List[Tuple[int, int]]:
    h_list, w_list = overlapping_grid_indices(height, width, patch_size, grid_r)
    return [(i, j) for i in h_list for j in w_list]


def data_transform(x):
    """[0, 1] -> [-1, 1]（官方同名函数）。"""
    return 2.0 * x - 1.0


def inverse_data_transform(x):
    """[-1, 1] -> [0, 1] 并裁剪（官方同名函数）。"""
    try:
        import torch  # noqa: PLC0415

        if isinstance(x, torch.Tensor):
            return torch.clamp((x + 1.0) / 2.0, 0.0, 1.0)
    except ImportError:
        pass
    return np.clip((x + 1.0) / 2.0, 0.0, 1.0)


def compute_alpha(beta, t):
    """
    计算 alpha_bar_t，形状 (B, 1, 1, 1)。官方同名函数。
    beta: torch.Tensor (T,)   t: torch.LongTensor (B,)
    """
    import torch  # noqa: PLC0415

    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)


# --------------------------------------------------------------------------- #
# Patch-based DDIM 采样主循环
# --------------------------------------------------------------------------- #
def generalized_steps_overlapping(
    x,
    x_cond,
    seq: List[int],
    model,
    betas,
    eta: float = 0.0,
    corners: Optional[List[Tuple[int, int]]] = None,
    patch_size: int = 64,
    patch_batch_size: int = 32,
    progress_cb=None,
    cancel_cb=None,
):
    """
    Patch-based DDIM 采样主循环。

    :param x:        初始噪声 (1, 3, H, W)，torch.Tensor
    :param x_cond:   条件图（退化图，值域 [0,1]） (1, 3, H, W)
    :param seq:      时间步序列，make_timestep_seq() 的返回值
    :param model:    DiffusionUNet
    :param betas:    torch.Tensor (T,)
    :param eta:      DDIM 的 eta，官方推理用 0
    :param corners:  patch 左上角坐标列表，corner_list() 的返回值
    :param patch_size:       patch 边长（= config.data.image_size）
    :param patch_batch_size: 一次并行推理多少个 patch（显存不够就调小）
    :param progress_cb:      progress_cb(step, total, preview_tensor_or_None)
    :param cancel_cb:        cancel_cb() -> bool，True 时应立即抛 RestoreCancelled
    :return: 最终 x0，torch.Tensor (1, 3, H, W)，值域 [-1, 1]
    """
    import torch  # noqa: PLC0415

    from ..base import RestoreCancelled

    if corners is None:
        h, w = x.shape[2], x.shape[3]
        corners = corner_list(h, w, patch_size, 16)

    total_steps = len(seq)
    seq_next = [-1] + list(seq[:-1])

    # 条件图变换到 [-1, 1]
    x_cond_neg = data_transform(x_cond)

    # 覆盖次数 mask (1, 1, H, W)
    x_grid_mask = torch.zeros_like(x[:, :1])
    for (hi, wi) in corners:
        x_grid_mask[:, :, hi:hi + patch_size, wi:wi + patch_size] += 1

    for step_idx, (t, next_t) in enumerate(zip(reversed(seq), reversed(seq_next))):
        # 取消检查
        if cancel_cb is not None and cancel_cb():
            raise RestoreCancelled("用户取消")

        at = compute_alpha(betas, torch.tensor([t], dtype=torch.long, device=x.device))
        at_next = compute_alpha(betas, torch.tensor([next_t], dtype=torch.long, device=x.device))

        # 1) 按 corners 切 patch，预测噪声并累加
        et_output = torch.zeros_like(x)
        for i in range(0, len(corners), patch_batch_size):
            batch_corners = corners[i:i + patch_batch_size]
            n_patches = len(batch_corners)
            # 组装 batch: (n_patches, 6, patch_size, patch_size)
            patch_batch = torch.zeros(n_patches, 6, patch_size, patch_size, device=x.device)
            for j, (hi, wi) in enumerate(batch_corners):
                patch_batch[j, :3] = x_cond_neg[:, :, hi:hi + patch_size, wi:wi + patch_size]
                patch_batch[j, 3:] = x[:, :, hi:hi + patch_size, wi:wi + patch_size]
            # 时间步广播到 (n_patches,)
            t_batch = torch.full((n_patches,), t, dtype=torch.long, device=x.device)
            # 模型推理
            with torch.no_grad():
                et_patch = model(patch_batch, t_batch)
            # 累加到整图
            for j, (hi, wi) in enumerate(batch_corners):
                et_output[:, :, hi:hi + patch_size, wi:wi + patch_size] += et_patch[j:j + 1]

        # 2) 按覆盖次数平均
        et = et_output / x_grid_mask

        # 3) DDIM 更新
        x0_t = (x - et * (1 - at).sqrt()) / at.sqrt()
        if eta == 0.0:
            # 确定性 DDIM
            xt_next = at_next.sqrt() * x0_t + (1 - at_next).sqrt() * et
        else:
            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et

        x = xt_next

        # 进度回调
        if progress_cb is not None:
            progress_cb(step_idx + 1, total_steps, x)

    return x
