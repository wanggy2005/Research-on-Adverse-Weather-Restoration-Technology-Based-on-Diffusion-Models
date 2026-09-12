"""
core.weatherdiff —— WeatherDiffusion 扩散模型实现

基于论文: WeatherDiffusion (TPAMI 2023)
官方代码: https://github.com/IGITUGraz/WeatherDiffusion

文件结构:
  - unet.py: DiffusionUNet 网络结构
  - sampling.py: Patch-based DDIM 采样算法
  - restorer.py: 推理编排（权重加载、预处理、后处理）

权重下载:
  WeatherDiff64 : https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff64.pth.tar
  WeatherDiff128: https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff128.pth.tar
"""
