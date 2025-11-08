## U-RNN 城市洪水高分辨率时空临近预报（《水文杂志 2025》）

U-RNN，首个潜在自回归时空架构（已在二维水动力学上验证），并采用 SWP 训练范式解决大规模时空建模挑战。

<p align="center"><img src="figs/pipeline.png" width="900"/></p>

- **标题**：U-RNN 城市洪水高分辨率时空临近预报
- **作者**：曹晓燕 等
- **论文**：Journal of Hydrology 2025（DOI 链接见英文版 `README.md`）

### 新闻 ✨
- （2025.05）在训练部分，我们提供了加速训练的建议：降低样本时空分辨率与减少 epoch 数量。
- （2025.05）即将发布 U-RNN 介绍的博客与视频（含论文与代码复现）。
- （2025.04）北京大学官方媒体报道我们的工作（[中文版链接](https://mp.weixin.qq.com/s/hbeWwhh_j46FiBgSIPL_jw)；[英文版链接](https://see.pkusz.edu.cn/en/info/1007/1156.htm)）。
- （2025.04）论文上线！
- （2025.03）U-RNN 被《水文杂志》接收！🎉🎉🎉
- （2024.12）UrbanFlood24 数据集发布为城市洪水建模新基准（项目页或百度云提取码：urnn）。

### 摘要
随着气候变化和快速城市化导致城市洪水风险加剧，迫切需要快速、准确、高分辨率的洪水流动时空临近预报模拟。尽管深度学习（DL）方法在提供此类信息方面前景广阔，但它们仍面临两个主要挑战：如何捕捉多尺度非线性时空相互依赖关系，以及如何在有限计算资源下优化长序列泛化能力。在此，我们提出一种新颖的类 U 型循环神经网络（U-RNN）来表征城市洪水的时空动态过程，并提出一种基于滑动窗口的预热（SWP）训练范式，以减少计算需求并增强全序列预测的泛化能力。对中国和英国四个高度城市化流域的案例研究表明，在遭遇重现期为 0.1 至 500 年的空间均匀或可变降雨事件时，U-RNN 在洪水范围、峰值深度和水位过程线方面实现了高预测精度，与最先进的水动力学模型相比，将 6 小时提前临近预报速度加快了 100 多倍。在空间 O (m) 量级和时间 O (min) 量级的高时空分辨率下，我们的方法精度也显著优于最先进的深度学习方法。对 U-RNN 的消融研究和对 SWP 的参数敏感性分析验证了每个模块的有效性。

## 0 临近预报性能

- 实测降雨预报可视化

<p align="center"><img src="figs/r100y_p0.5_d3h.gif" width="600"/></p>

- 测试集上的定量评估

<p align="center"><img src="figs/overall_results.png" width="600"/></p>

## 1 安装

下载代码仓库：

```bash
git clone https://github.com/holmescao/U-RNN
```

安装 Python 依赖项（我们在兼容的 NVIDIA GeForce RTX 4090 上使用 Python 3.8 和 PyTorch 2.0.0）：

```bash
conda create -n urnn python=3.8
conda activate urnn
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
cd U-RNN # 进入代码根目录
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

其他所需依赖项：

- CUDA 11.8
- cuDNN 8.9.0

注意：其他版本也可以运行，只要 PyTorch、GPU、CUDA 和 cuDNN 版本兼容即可。

## 2 数据准备

通过官方项目页面或百度云（提取码：urnn）下载城市洪水数据集，解压并放入路径（`<U-RNN_HOME>/data`）。

数据集结构如下：

```text
data
└── urbanflood24
    ├── test
    │   └── flood
    │       └── location16
    │           ├── G1135_intensity_103
    │           │   ├── flood.npy
    │           │   └── rainfall.npy
	...
        └── geodata
            └── location16
                ├── absolute_DSM.npy
                ├── impervious.npy
                └── manhole.npy
    ...  
    └── train
        ├── flood
        │   └── location16
        │       ├── G1135_intensity_117
        │       │   ├── flood.npy
        │       │   └── rainfall.npy
	...
        └── geodata
            └── location16
                ├── absolute_DSM.npy
                ├── impervious.npy
                └── manhole.npy
    ...
```

## 3 训练好的模型权重

`location16` 的训练好的 U-RNN 模型权重已提供（其他位置需要训练），可在谷歌云或百度云（提取码：urnn）获取。请下载并放入路径：

```text
<U-RNN_HOME>/exp/20240202_162801_962166/save_model/checkpoint_939_0.000205376.pth.tar
```

## 4（可选）训练

注意：考虑到有限的计算资源，建议将数据集的时空分辨率从当前的 500×500×360 采样降低到 200×200×36，这可以显著加快训练过程（使用单个 4090 GPU，一天内即可完成训练）。此外，减少 epoch 数量也是提高训练速度的直接方法。例如，从 1000 个 epoch 减少到 300 个，仍然可以达到较高的精度。

在终端中运行以下命令：

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main.py --device 0 --batch_size 1 --seq_num 28 --use_checkpoint
```

实验结果的根目录在路径（`<U-RNN_HOME>/exp`）中。该脚本将生成一个时间戳作为实验名称，并创建一个文件夹用于保存相关文件，例如 `exp/20240202_162801_962166`。

## 5 推理

1) 打开 `test.py`，找到 `if __name__ == "__main__":` 的下一行，将 `timestamp` 变量设置为你训练的模型实验名称，例如：

```python
timestamp = "20240202_162801_962166"
```

2) 在终端中运行：

```bash
python test.py
```

结果将保存在路径（`<U-RNN_HOME>/exp/<your_expr_name>/figs/`）中。对于每个降雨事件，你将获得推理结果的可视化，如下所示：

<p align="center"><img src="figs/demo.png" width="800"/></p>

## 6（可选）使用 TensorRT 进行推理

将模型从 PyTorch 转换为 TensorRT（我们使用 TensorRT 10.0.0.6）。

1) 打开 `urnn_to_tensorrt.py`，在 `if __name__ == "__main__":` 的下一行设置 `timestamp`：

```python
timestamp = "20240202_162801_962166"
```

2) 在终端中运行：

```bash
python urnn_to_tensorrt.py
```

你将在路径（`<U-RNN_HOME>/exp/<your_expr_name>/tensorrt/`）中得到一个 TensorRT 模型 `URNN.trt`。

3) 使用 TensorRT 进行推理：

```bash
python test.py --trt
```

结果将保存在路径（`<U-RNN_HOME>/exp/<your_expr_name>/figs/`）中。

## 7 引用

如果您发现这个项目有用，请考虑引用我们的论文。谢谢！

```text
@article{cao2025u,
  title={U-RNN high-resolution spatiotemporal nowcasting of urban flooding},
  author={Cao, Xiaoyan and Wang, Baoying and Yao, Yao and Zhang, Lin and Xing, Yanwen and Mao, Junqi and Zhang, Runqiao and Fu, Guangtao and Borthwick, Alistair GL and Qin, Huapeng},
  journal={Journal of Hydrology},
  pages={133117},
  year={2025},
  publisher={Elsevier}
}

@misc{cao2024supplementary,
  author       = {Cao, Xiaoyan and Wang, Baoying and Qin, Huapeng},
  title        = {Supplementary data of “U-RNN high-resolution spatiotemporal nowcasting of urban flooding”},
  year         = {2024},
  publisher    = {figshare},
  note         = {Dataset}
}
```
