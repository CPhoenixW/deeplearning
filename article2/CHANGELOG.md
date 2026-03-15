# 更新日志 (CHANGELOG)

## 联邦鲁棒聚合框架 - 两阶段 AE-SVDD

本项目实现了一个针对联邦学习的鲁棒聚合框架，通过两阶段 AutoEncoder-SVDD（Support Vector Data Description）机制检测并过滤恶意客户端，提升全局模型在拜占庭攻击下的鲁棒性。

---

## [v0.2.0] - 2026-03-13

### 新增
- `move/` 模块：重构后的核心实现，包含优化后的聚合策略

### 变更
- **Phase 1 预热轮次**：从 20 轮延长至 50 轮，给予 AutoEncoder 更充分的预训练时间
- **动态 λ 权重调度**：引入 `LAMBDA_START=0.1` 到 `LAMBDA_END=1.0` 的退火机制，SVDD Loss 权重随训练进度动态增长
- **动态动量 β**：新增球心更新的动态动量参数，从 0 线性增长至 0.9，提升后期训练稳定性
- **高斯噪声强度**：攻击客户端的噪声标准差 `sigma` 从 0.5 调整为 0.6，提高攻击强度以更好测试防御效果

### 移除
- 移除 SVDD 阶段的梯度裁剪（`clip_grad_norm_`），简化训练流程
- 移除 Phase 1 的显式学习率重置逻辑

### 优化
- 权重计算逻辑简化，提升代码可读性

---

## [v0.1.0] - 2026-03-10

### 核心功能
- **两阶段 AE-SVDD 聚合框架**
  - Phase 1（Round 1-20）：AutoEncoder 预训练 + 均匀权重 FedAvg
  - Phase 2（Round 21+）：SVDD 过滤 + 软权重聚合
  
- **全局模型**：ResNet-18（适配 CIFAR-10）

- **异常检测模型**
  - Encoder：BN特征 → 128维潜在空间（无偏置约束）
  - Decoder：潜在向量 → BN特征重建
  - SVDD 球心动态更新与硬截断过滤

### 客户端类型
| 类型 | 描述 |
|------|------|
| `BenignClient` | 良性客户端，标准 SGD 本地训练 |
| `GaussianNoiseClient` | 高斯投毒，对模型参数添加 N(0, σ²) 噪声 |
| `LabelFlippingClient` | 标签翻转攻击，y → 9-y |
| `SignFlippingClient` | 反向梯度攻击，上传 global - scale × Δ |

### 超参数配置
- 客户端数量：10（默认 7 良性 + 3 恶意）
- 通信轮次：300 轮
- 潜在空间维度：128
- 温度退火：T 从 5.0 → 0.5
- 边界系数退火：k 从 3.0 → 1.0
- 本地学习率：0.1（SGD with momentum=0.9）

### 监控与评估
- 实时输出类 nvidia-smi 风格的监控面板
- 记录指标：测试准确率、TPR（恶意拒绝率）、FPR（良性误拒率）
- 自动生成带时间戳的训练日志

---

## 项目结构

```
article2/
├── maintain/          # v0.1.0 稳定版本
│   ├── main.py        # 主入口
│   ├── models.py      # ResNet-18 + AutoEncoder
│   ├── server.py      # 两阶段 AE-SVDD 聚合服务器
│   ├── clients.py     # 良性/恶意客户端实现
│   ├── dataset.py     # CIFAR-10 数据加载与划分
│   ├── utils.py       # BN 特征提取工具
│   └── requirements.txt
├── move/              # v0.2.0 开发版本（含最新优化）
│   └── ...            # 结构同上
├── logs/              # 训练日志（按时间戳命名）
└── CHANGELOG.md       # 本文件
```

---

## 运行方式

```bash
# 激活虚拟环境
source dl/bin/activate

# 运行默认配置（AE-SVDD + 高斯噪声攻击）
cd maintain  # 或 cd move
python main.py

# 切换攻击类型
# 修改 main.py 中的 attack_type 参数：
# "gaussian_noise" | "label_flipping" | "sign_flipping"
```

---

## 参考文献

- FedAvg: Communication-Efficient Learning of Deep Networks from Decentralized Data
- Deep SVDD: Deep One-Class Classification
- Byzantine-Robust Distributed Learning
