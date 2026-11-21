##  Attention is all you need
Simple Transformer Implemented by Pytorch.
基于谷歌原论文的transformer实现，在舍弃了数据预处理和工程优化等内容，旨在帮助初学者快速复现并理解transformer和pytorch的内容架构，同时聚焦于自己复现时遇到的各种问题，尽量详细的做到适合中国宝宝体质的无障碍理解。

PS:感谢[ngolin](https://github.com/ngolin)提供了框架和数据集，本项目使用的数据来自于[AttentionIsAllYouNeed](https://github.com/ngolin/AttentionIsAllYouNeed)

## 快速开始
配置环境
```bash
    pip install -r requirements.txt
```
快速开始
训练模型
```bash
    $ python main.py
```
利用训练好的 best_model.pt进行测试
```bash
    $ python test.py
```

## 在服务器上运行（CPU/GPU）
本仓库已适配无显示的服务器环境：
- 已将 Matplotlib 切换为非交互后端（Agg），训练/测试会将损失曲线保存为图片文件（training_loss.png / training_loss_test.png）。
- 加载模型使用 map_location，避免 GPU/CPU 不匹配导致的加载失败。

1) 基础依赖
- Python 3.9+（建议与你本地一致）
- 安装依赖
```bash
pip install -r requirements.txt
```

2) PyTorch 安装说明
- CPU 服务器（最简单，适合无 GPU）
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
- GPU 服务器（根据服务器 CUDA 版本选择）
参考 https://pytorch.org/get-started/locally/ 选择匹配 CUDA 的安装命令。例如（以 CUDA 12.x 为例）：
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
```
注意：requirements.txt 中的版本是参考，若与你的服务器 CUDA 不匹配，请以官网命令为准安装可用版本。

3) 数据与目录
- 数据文件已包含在 `./dataset/cmn-eng.txt`。
- 训练过程中会在项目根目录保存 `best_model.pt` 与损失曲线图片。

4) 先做一次环境冒烟测试（强烈推荐）
```bash
python smoke_test.py
```
输出包含 batch 形状、一次前后向的损失和用时，看到“Smoke test OK”即可说明环境可跑。

5) 启动训练/测试
- 训练
```bash
python main.py
```
输出日志中会打印每个 epoch 的平均 loss，并在 loss 改善时更新 `best_model.pt`，训练结束会生成 `training_loss.png`。
- 使用最佳模型测试
```bash
python test.py
```


6) 后台运行（可选）
- Linux 常用：tmux/screen/nohup。例如：
```bash
nohup python main.py > train.out 2>&1 &
```
- 或在 tmux 内启动：
```bash
tmux new -s tf
python main.py
```

7) 常见问题
- 找不到 CUDA 或驱动不匹配：请用 CPU 版本的 torch 先跑通，或按 PyTorch 官网选择正确 CUDA 版本安装。
- 服务器无显示导致报错：本项目已将绘图改为保存到文件，不需要显示。
- best_model.pt 在 CPU/GPU 间加载错误：我们已使用 map_location 自动适配当前设备。

## 文件说明
main.py：训练代码
model.py：模型结构与具体实现
best_model.pt: 训练过程中得到的最好的模型
test.py 实验模型训练效果（中译英）
## 实验结果
![alt text](image.png)

## 对论文中Transformer变体的复现
之前完整 WMT14 En-De 训练集的实际大小是：

```text
4,508,785 条句对
```

在默认 `batch_size=32` 下：

```text
一个 epoch 的 batch 数 = ceil(4,508,785 / 32) = 140,900
```

你之前看到的进度条：

```text
70/17613
```

这里的 `17613` 不是 batch 数，而是 **optimizer step 数**，因为默认：

```text
accum_steps = 8
```

```text
140900 // 8 = 17612 optimizer steps
```

所以：

| 项目 | 数量 |
|---|---:|
| 训练句对 | 4,508,785 |
| batch_size | 32 |
| 每个 epoch batch 数 | 140,900 |
| accum_steps | 8 |
| 每个 epoch optimizer steps | 17,612 |
| 1 个 epoch 的 5% | 7,045 batches |
| 换算成 optimizer steps | 约 881 steps |

如果你想按 5% epoch 打一次日志，可以设置：

```bash
--log-batch-interval 7045
```

当前默认的 `--log-batch-interval 1000` 大约是每 `0.71%` epoch 打一次日志。

-
​
 单步batch容量：src≈25k token，tgt≈25k token
​
> We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [3], which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [38]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.
中文： 我们在标准 WMT2014 英德数据集（约450万句对）上训练，句子用BPE编码，源端目标端共享约37000词表；英法任务用更大的WMT2014英法数据集（3600万句），词表32000。按相近序列长度打包句对。每个训练batch大约包含25000个source token + 25000个target token。
​
> ✅ 论文这里的 batch：token 容量，不是固定句子数量；一个 batch = 一次 optimizer step（无梯度累积）

￼
​
​
￼
5.2 Hardware and Schedule（总 step 数原文）
​
> We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models, (described on the bottom line of table 3), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).
中文： 单台8卡P100训练。Base模型每step约0.4s，总共训练100,000 step（12小时）；Big模型每step约1s，总共训练300,000 step（3.5天）。
​
> ✅ 这里的 step = optimizer step（参数更新次数）

￼
​
​
￼
## 5.3 Optimizer（warmup step 原文）
​
> We used the Adam optimizer [20] with 𝛽1=0.9, 𝛽2=0.98 and 𝜖=10−9. We varied the learning rate over the course of training, according to the formula: 
𝑙𝑟𝑎𝑡𝑒=𝑑−0.5
𝑚𝑜𝑑𝑒𝑙⋅min(𝑠𝑡𝑒𝑝𝑛𝑢𝑚−0.5,𝑠𝑡𝑒𝑝𝑛𝑢𝑚⋅𝑤𝑎𝑟𝑚𝑢𝑝𝑠𝑡𝑒𝑝𝑠−1.5)
￼
 This corresponds to increasing the learning rate linearly for the first 𝑤𝑎𝑟𝑚𝑢𝑝𝑠𝑡𝑒𝑝𝑠 training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. **We used 𝑤𝑎𝑟𝑚𝑢𝑝𝑠𝑡𝑒𝑝𝑠=4000.**
中文： Adam，𝛽1=0.9,𝛽2=0.98,𝜖=10−9；学习率按上面公式调度：前 `\(warmup\_steps\)` 线性升lr，之后按 step_num 的平方根倒数衰减。**(warmup_steps) = 4000**。




## 参考资料
[论文原文](https://arxiv.org/abs/1706.03762)
[论文精讲](https://www.youtube.com/watch?v=nzqlFIcCSWQ)
[AttentionIsAllYouNeed](https://github.com/ngolin/AttentionIsAllYouNeed)
