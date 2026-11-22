# ⚛️ NQS Cloud Workflow (JAX / NetKet + Mini-GPT Wavefunction)

专为**零基础快速上手神经量子态（Neural Quantum States, NQS）**且**不依赖本地电脑常开**设计的云端计算与监控脚手架：

* **免费算力端（Google Colab T4 GPU）**：基于 `JAX` + `NetKet 3`，内置 **Causal GPT 自回归波函数**与经典 **RBM**，首次运行自动计算精确对角化（Exact Diagonalization）标准答案，1 分钟复现能量收敛曲线。
* **防断电/防掉线双保险（Google Drive + W&B Artifacts）**：每 50 步原子化保存波函数参数；哪怕本地断电、Colab 掉线，下次打开点 `Run all` 自动从断点接着算。
* **手机随时看曲线 + 7×24 云端哨兵（W&B + Google Cloud 永久免费 `e2-micro` VM）**：不跑耗内存易封号的浏览器爬虫，而是通过轻量 API 轮询心跳（内存占用 `< 40 MB`），一旦检测到 Colab 掉线、能量发散（`NaN`）或计算完成，自动推送通知到手机。

---

## 🧠 1 分钟搞懂：为什么“NQS 用 GPT 算得快”？

在量子多体物理中，$N$ 个自旋的量子态有 $2^N$ 个分量，传统方法很快算爆内存。**NQS** 用神经网络输入一组自旋构型 $(s_1, s_2, \dots, s_N)$，直接输出波函数对数幅值 $\log \psi(s)$。

这里其实有两个层面的加速（本项目同时为你配齐了）：
1. **硬件加速（GPU + JAX）**：Colab 免费提供的 T4 GPU 配合 Google 的 `JAX` 自动编译（XLA），让矩阵运算比普通 CPU 快数十倍。
2. **算法加速（GPT 自回归波函数 + 精确直接采样）**：
   * 传统 NQS（如受限玻尔兹曼机 `RBM`）必须靠马尔可夫链（MCMC）一步步随机翻转自旋来采样，不仅要丢弃预热步数，还容易卡在局部构型里。
   * **GPT 式自回归波函数**（见 [src/nqs/models.py](src/nqs/models.py) 中的 `CausalGPTWaveFunction`）把波函数拆解为因果条件概率的乘积：
     $$\psi(s_1, \dots, s_N) = \prod_{i=1}^{N} \psi(s_i \mid s_1, \dots, s_{i-1})$$
   * 就像 GPT 逐个生成文字一样，它可以利用 `ARDirectSampler` 在 GPU 上**一次性无偏、零自相关地直接生成上千个独立量子构型**！

---

## 📁 项目目录结构

```text
NQS/
├── configs/
│   ├── tfim_1d_gpt.yaml        # 【首跑推荐】1D 横场伊辛临界点 (L=12) + Mini-GPT 波函数（约 1 分钟跑完，含精确解真值）
│   ├── tfim_1d_rbm.yaml        # 【经典对照】Carleo & Troyer (Science 2017) RBM + 量子自然梯度 SR
│   └── heisenberg_1d_gpt.yaml  # 【进阶实验】1D 反铁磁海森堡链 (L=12) + 复相位 GPT 波函数
├── src/nqs/
│   ├── config.py               # 实验 YAML 配置解析与参数合法性校验
│   ├── physics.py              # 量子自旋哈密顿量构建与 Lanczos 精确对角化（标准答案求解器）
│   ├── models.py               # CausalGPTWaveFunction (Mini-GPT)、ARNNDense 与 RBM 实现
│   ├── checkpoint.py           # 原子化断点续训（同时支持 Google Drive 与 W&B 云端备份）
│   └── train.py                # VMC 训练主循环、精确误差与 V-score 计算、心跳上报与收敛绘图
├── notebooks/
│   └── colab_runner.ipynb      # Google Colab 一键运行工作台
├── monitor/
│   ├── gcp_watchdog.py         # 跑在 GCP 免费 e2-micro 虚拟机上的 7×24 哨兵脚本
│   ├── setup_gcp_vm.sh         # GCP 虚拟机一键配置 systemd 开机自启脚本
│   └── monitor_config.example.yaml
└── docs/
    └── WORKFLOW_GUIDE.md       # 保姆级全流程配置手册（GitHub -> Colab -> W&B -> GCP 免费 VM）
```

---

## 🚀 快速开始

请直接阅读 **[docs/WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md)**，按顺序完成 4 步即可跑通第一个基准算例并挂上云端 24 小时监控。
