# weights/ 权重目录

本目录**不入 Git**（见 .gitignore），只保留本说明文件。

## 一键下载

```bash
python scripts/download_weights.py          # 下载 WeatherDiff64（默认，推荐）
python scripts/download_weights.py --list   # 查看可下载的权重
python scripts/download_weights.py --all    # 两个都下
```

## 官方权重直链（TU Graz 服务器，无需 Google Drive）

| 文件名 | 下载地址 | 说明 |
|---|---|---|
| WeatherDiff64.pth.tar | https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff64.pth.tar | patch=64，configs 默认用它 |
| WeatherDiff128.pth.tar | https://igi-web.tugraz.at/download/OzdenizciLegensteinTPAMI2023/WeatherDiff128.pth.tar | patch=128，更慢更吃显存，需把 `data.image_size` 改成 128 |

注意：官方只发布了**统一的多天气模型**，没有分别针对雨/雾/雪的三份权重。
因此 `configs/rain.yaml`、`haze.yaml`、`snow.yaml` 共用同一份权重文件。

## 下载后自检

```bash
python scripts/check_weights.py                 # 打印 checkpoint 结构
python scripts/check_weights.py --try-build     # 真正构建模型并加载（需 unet.py 已实现）
```

## 备选权重（主方案效果/速度不满意时）

| 方案 | 仓库 |
|---|---|
| DiffUIR (CVPR2024) 统一复原 | https://github.com/iSEE-Laboratory/DiffUIR |
| Restormer（非扩散对照） | https://github.com/swz30/Restormer |
| TransWeather（多天气单模型） | https://github.com/jeya-maria-jose/TransWeather |
| DehazeFormer（纯去雾兜底） | https://github.com/IDKiro/DehazeFormer |

下载到本目录后，新建一个 `configs/xxx.yaml` 指向它即可，无需改动界面代码。
