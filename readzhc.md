## 项目简介

U-RNN 是一种面向城市洪水高分辨率时空临近预报的深度学习架构。它提出了首个“潜在自回归”的时空建模思路，并结合基于滑动窗口的预热训练范式（SWP），在显著降低计算开销的同时，提升长序列预测的泛化能力。相较于传统二维水动力模型，U-RNN 在保证精度的前提下可将 6 小时提前的临近预报速度提升百倍以上，并在 O(m) 空间量级、O(min) 时间量级下取得领先性能。

适用场景：快速生成城市洪水的范围、深度与水位过程线等时空预报结果，用于应急研判、方案比选与科学研究。

---

## 功能模块概览

- 模型结构（`src/lib/model/networks/`）
  - `encoder.py` / `decoder.py`：U 形结构的时空特征编码与重建。
  - `ConvRNN.py`：卷积循环单元，建模局地-全局时空依赖。
  - `head/flood_head.py`、`head/network_blocks.py`：任务头与网络积木。
  - `losses.py`：训练损失定义。
  - `model.py`、`net_params.py`、`utils.py`：模型组装、超参与辅助工具。
- 数据集（`src/lib/dataset/`）
  - `Dynamic2DFlood.py`：数据加载、序列切片与批处理。
  - `train.txt` / `test.txt`：样本索引示例。
- 训练与早停（`src/lib/model/earlystopping.py`）
  - 提供监控指标的早停逻辑，防止过拟合。
- 运行脚本
  - `main.py`：分布式/单卡训练入口。
  - `test.py`：离线推理与可视化。
  - `urnn_to_tensorrt.py`：PyTorch → TensorRT 转换与推理。
  - `config.py`：全局配置（如路径、默认参数）。
  - `src/lib/utils/`：通用工具（日志、分布式、记录、Torch 实用函数等）。

---

## 环境与安装

建议环境：Python 3.8，PyTorch 2.0.0，CUDA 11.8，cuDNN 8.9.0（其他兼容组合亦可）。

```bash
conda create -n urnn python=3.8
conda activate urnn
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
cd U-RNN
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

若网络受限，可优先使用国内镜像源安装依赖。

---

## 数据准备

1) 从项目页面或网盘获取 UrbanFlood24 数据集，并解压至 `<U-RNN_HOME>/data`：

```text
<U-RNN_HOME>/data
└── urbanflood24
    ├── train
    │   ├── flood/.../flood.npy, rainfall.npy
    │   └── geodata/.../absolute_DSM.npy, impervious.npy, manhole.npy
    └── test
        ├── flood/.../flood.npy, rainfall.npy
        └── geodata/.../absolute_DSM.npy, impervious.npy, manhole.npy
```

2) 预训练权重（可选，免训练直接推理）。以 `location16` 为例，将下载的权重放置到：

```text
<U-RNN_HOME>/exp/20240202_162801_962166/save_model/checkpoint_939_0.000205376.pth.tar
```

---

## 训练（可选）

在算力有限时，建议将样本从 500×500×360 采样降至约 200×200×36，并相应减少 epoch（如从 1000 降至 300），可在单张 4090 上一天内完成训练。

单卡示例（分布式后端仍可开启 1 进程）：

```bash
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 \
  main.py --device 0 --batch_size 1 --seq_num 28 --use_checkpoint
```

运行后会在 `exp/` 下生成时间戳实验目录（如 `exp/20240202_162801_962166`），用于保存配置、日志、模型与可视化。

---

## 推理与可视化

1) 打开 `test.py`，在 `if __name__ == "__main__":` 下方设置 `timestamp` 为你的实验目录名，例如：

```python
timestamp = "20240202_162801_962166"
```

2) 运行推理：

```bash
python test.py
```

3) 结果将保存至：

```text
<U-RNN_HOME>/exp/<your_expr_name>/figs/
```

目录内包含每个降雨事件对应的时空可视化图像/动图，便于快速对比评估。

---

## 使用 TensorRT（可选）

1) 转换：

```python
# 在 urnn_to_tensorrt.py 中设置
timestamp = "20240202_162801_962166"
```

```bash
python urnn_to_tensorrt.py
```

将在：

```text
<U-RNN_HOME>/exp/<your_expr_name>/tensorrt/URNN.trt
```

2) TensorRT 推理：

```bash
python test.py --trt
```

输出同样保存至 `exp/<your_expr_name>/figs/`。

---

## 典型文件结构速览

```text
U-RNN/
  ├─ main.py                # 训练入口
  ├─ test.py                # 推理与可视化
  ├─ urnn_to_tensorrt.py    # TensorRT 转换/推理
  ├─ config.py              # 全局配置
  ├─ src/
  │  └─ lib/
  │     ├─ dataset/
  │     │  └─ Dynamic2DFlood.py
  │     ├─ model/
  │     │  ├─ earlystopping.py
  │     │  └─ networks/
  │     │     ├─ encoder.py / decoder.py / ConvRNN.py
  │     │     ├─ head/
  │     │     │  ├─ flood_head.py
  │     │     │  └─ network_blocks.py
  │     │     ├─ losses.py / model.py / net_params.py / utils.py
  │     └─ utils/           # 日志、分布式、记录、通用工具
  └─ exp/                   # 实验输出（自动创建）
```

---

## 常见问题（FAQ）

- 显存不足？
  - 将分辨率/序列长度下采样；减小 `batch_size`；启用 `--use_checkpoint`；使用混合精度（若已支持）。
- 训练太慢？
  - 先用低分辨率与较少 epoch 调参，收敛后再逐步提高分辨率或时长。
- 版本不匹配？
  - 保持 PyTorch、CUDA、驱动、cuDNN 版本相互兼容；必要时更换对应的 CUDA Toolkit 与轮子。

---

如需进一步帮助或学术引用，请参考根目录 `README.md` / `README_CN.MD` 中的安装说明、数据链接与引用信息。


