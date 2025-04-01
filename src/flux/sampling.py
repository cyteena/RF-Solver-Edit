import math
from typing import Callable

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    """
    准备模型输入数据，包括图像、图像ID、文本、文本ID和向量。

    参数:
    t5 (HFEmbedder): T5嵌入器，用于将文本转换为嵌入向量。
    clip (HFEmbedder): CLIP嵌入器，用于将文本转换为向量。
    img (Tensor): 输入的图像张量。
    prompt (str | list[str]): 输入的文本提示，可以是单个字符串或字符串列表。

    返回:
    dict[str, Tensor]: 包含准备好的图像、图像ID、文本、文本ID和向量的字典。
    """
    # 获取图像的批次大小、通道数、高度和宽度
    bs, c, h, w = img.shape
    # 如果批次大小为1且提示不是字符串，则更新批次大小为提示列表的长度
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    # 重新排列图像张量的维度
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    # 如果图像的批次大小为1且更新后的批次大小大于1，则复制图像以匹配批次大小
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    # 初始化图像ID张量
    img_ids = torch.zeros(h // 2, w // 2, 3)
    # 在图像ID的第二个通道(channel)上添加高度索引
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    # 在图像ID的第三个通道(channel)上添加宽度索引
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    # 复制图像ID以匹配批次大小
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    # 如果提示是单个字符串，则将其转换为包含单个字符串的列表
    if isinstance(prompt, str):
        prompt = [prompt]
    # 使用T5嵌入器将提示转换为文本嵌入向量
    txt = t5(prompt)
    # 如果文本嵌入向量的批次大小为1且更新后的批次大小大于1，则复制文本嵌入向量以匹配批次大小
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    # 初始化文本ID张量
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    # 使用CLIP嵌入器将提示转换为向量
    vec = clip(prompt)
    # 如果向量的批次大小为1且更新后的批次大小大于1，则复制向量以匹配批次大小
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
