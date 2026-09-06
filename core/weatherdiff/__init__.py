"""
core.weatherdiff —— 扩散模型实现（模型组负责）

需要移植的官方代码: https://github.com/IGITUGraz/WeatherDiffusion
  官方 models/unet.py          -> 本包 unet.py       (类名保持 DiffusionUNet)
  官方 utils/sampling.py       -> 本包 sampling.py   (patch-based DDIM 采样)
  官方 models/__init__.py 的推理部分 -> 本包 restorer.py

权重（官方只公开统一的多天气模型，没有分任务权重）:
  WeatherDiff64 : https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff64.pth.tar
  WeatherDiff128: https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff128.pth.tar
"""
