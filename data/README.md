# data/ 数据集目录

本目录**不入 Git**（见 .gitignore），只保留本说明文件。
数据集需自行下载整理，不要提交到仓库。

## 约定的目录结构

批量评测要求"退化图目录"与"GT 目录"分开存放，文件名主干一致：

```
data/
├── raindrop/            # 雨滴 (RainDrop test_a)
│   ├── input/
│   └── gt/
├── rainfog/             # 雨+雾 (Outdoor-Rain test1)
│   ├── input/
│   └── gt/
├── snow/                # 雪 (Snow100K-S / Snow100K-L 测试子集)
│   ├── input/
│   └── gt/
├── haze/                # 纯雾 (RESIDE SOTS-outdoor 子集)
│   ├── input/
│   └── gt/
└── real/                # 真实网络图，无 GT，只算 NIQE/BRISQUE
    └── input/
```

文件名可以带后缀，程序会自动配对，例如
`0_rain.png` <-> `0_clean.png`、`img01.jpg` <-> `img01.png` 都能配上
（配对规则见 `core/image_utils.py` 的 `match_pairs`）。

## 数据集来源

| 天气 | 数据集 | 地址 |
|---|---|---|
| 雨滴 | RainDrop | https://github.com/rui1996/DeRaindrop |
| 雨+雾 | Outdoor-Rain | https://github.com/liruoteng/HeavyRainRemoval |
| 雪 | Snow100K | https://sites.google.com/view/yunfuliu/desnownet |
| 纯雾 | RESIDE | https://sites.google.com/view/reside-dehaze-datasets |
| 三合一 | AllWeather (TransWeather 整理版) | https://github.com/jeya-maria-jose/TransWeather |

## 规模建议

每类**只取 50~200 对**测试图就够出指标了，不要下训练集（几十 GB）。
FID 需要每个目录至少 50 张，否则数值不可靠。

## 没有数据集时怎么开发

`core/image_utils.make_demo_image()` 能生成带"雨纹噪声"的合成图，
`scripts/smoke_test.py` 就是用它跑通全链路的，不需要任何真实数据。
