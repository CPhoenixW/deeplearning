# 联邦学习恶意参与方检测：详细实验方案

**设计时间**: 2026-03-25
**基于**: AE-SVDD方法 + 最新论文(FLARE, GShield, FedTrident)
**目标**: 全面评估Byzantine检测性能

---

## 1. 实验总体设计

### 1.1 实验目标

1. **验证AE-SVDD有效性**: 与基线方法对比
2. **评估攻击鲁棒性**: 多种攻击类型
3. **测试可扩展性**: 不同客户端数量
4. **分析非IID影响**: 数据异构性
5. **对标最新方法**: FLARE, GShield性能

### 1.2 实验矩阵

```
攻击类型 × 攻击率 × 客户端数 × 数据分布 × 防御方法
```

**维度1: 攻击类型** (6种)
- 标签翻转 (Label Flipping)
- 高斯噪声 (Gaussian Noise)
- 符号翻转 (Sign Flipping)
- ALIE (A Little Is Enough)
- Statistical Mimicry (新型)
- 无攻击 (Baseline)

**维度2: 攻击率** (5档)
- 10% (1/10 clients)
- 20% (2/10 clients)
- 30% (3/10 clients)
- 40% (4/10 clients)
- 50% (5/10 clients)

**维度3: 客户端数** (4档)
- 10 clients (小规模)
- 20 clients (中规模)
- 50 clients (大规模)
- 100 clients (超大规模, 对标FLARE)

**维度4: 数据分布** (2种)
- IID (独立同分布)
- Non-IID (Dirichlet α=0.5)

**维度5: 防御方法** (5种)
- FedAvg (无防御)
- Trimmed Mean (基础)
- Krum (几何)
- AE-SVDD (你的方法)
- FLARE (最新, 如可实现)

---

## 2. 详细实验配置

### 2.1 基础配置 (Exp-Base)

**数据集**: CIFAR-10
**模型**: ResNet18
**客户端**: 10 (7良性, 3恶意)
**轮数**: 300
**攻击**: 标签翻转 100%

**超参数**:
```python
# 联邦学习
num_clients = 10
num_benign = 7
total_rounds = 300
local_epochs = 1
batch_size = 32
client_lr = 0.1
client_momentum = 0.9
client_weight_decay = 5e-4

# AE-SVDD
latent_dim = 64
ae_lr = 1e-3
ae_weight_decay = 1e-6
ae_grad_clip = 1.0
phase1_rounds = 50
buffer_capacity = 500

# SVDD
svdd_warmup_rounds = 100
center_ema_decay = 0.9
tau_multiplier = 3.0
softweight_T_start = 5.0
softweight_T_end = 0.5
svdd_grad_clip = 1.0
svdd_recon_lambda = 0.1

# 其他
seed = 42
device = "cuda"
```

**预期结果**:
- 无攻击精度: ~92%
- 有攻击(无防御): ~10-20%
- AE-SVDD: ~85-90%

---

### 2.2 攻击类型实验 (Exp-Attack)

#### 2.2.1 标签翻转 (Label Flipping)

**配置**:
```python
attack_type = "label_flipping"
num_malicious = 3
flip_ratio = 1.0  # 100%翻转
```

**实现**:
```python
def label_flipping_attack(y, num_classes=10):
    return (num_classes - 1) - y
```

**预期**: 
- 无防御: 精度 < 20%
- AE-SVDD: TPR > 90%, 精度 > 85%

#### 2.2.2 高斯噪声 (Gaussian Noise)

**配置**:
```python
attack_type = "gaussian_noise"
gaussian_sigma = 0.5  # 标准差
```

**实现**:
```python
def gaussian_noise_attack(params, sigma=0.5):
    noise = torch.randn_like(params) * sigma
    return params + noise
```

**预期**:
- 无防御: 精度 < 30%
- AE-SVDD: TPR > 85%, 精度 > 80%

#### 2.2.3 符号翻转 (Sign Flipping)

**配置**:
```python
attack_type = "sign_flipping"
sign_flip_scale = 1.0
```

**实现**:
```python
def sign_flipping_attack(global_params, local_params, scale=1.0):
    delta = local_params - global_params
    return global_params - scale * delta
```

**预期**:
- 无防御: 精度 < 40%
- AE-SVDD: TPR > 80%, 精度 > 75%

#### 2.2.4 ALIE (A Little Is Enough)

**配置**:
```python
attack_type = "alie"
alie_threshold = 0.5  # 选择阈值
```

**实现**:
```python
def alie_attack(gradients, threshold=0.5):
    # 选择最小的梯度方向
    sorted_grads = torch.sort(gradients, dim=0)[0]
    idx = int(len(gradients) * threshold)
    return sorted_grads[idx]
```

**预期**:
- 无防御: 精度 < 50%
- AE-SVDD: TPR > 75%, 精度 > 70%

#### 2.2.5 Statistical Mimicry (新型自适应攻击)

**配置**:
```python
attack_type = "statistical_mimicry"
mimicry_alpha = 0.5  # 混合比例
mimicry_drift = 0.01  # 漂移强度
```

**实现**:
```python
def statistical_mimicry_attack(honest_grad, noise_scale=0.1, drift=0.01):
    noise = torch.randn_like(honest_grad) * noise_scale
    drift_term = torch.randn_like(honest_grad) * drift
    return 0.5 * honest_grad + 0.5 * noise + drift_term
```

**预期**:
- 无防御: 精度 < 45%
- AE-SVDD: TPR > 70%, 精度 > 65%

---

### 2.3 攻击率实验 (Exp-Rate)

**配置**: 标签翻转, 10个客户端

| 攻击率 | 恶意客户端 | 配置 | 预期TPR | 预期精度 |
|--------|-----------|------|---------|---------|
| 10% | 1/10 | num_benign=9 | >95% | >90% |
| 20% | 2/10 | num_benign=8 | >92% | >88% |
| 30% | 3/10 | num_benign=7 | >90% | >85% |
| 40% | 4/10 | num_benign=6 | >85% | >80% |
| 50% | 5/10 | num_benign=5 | >80% | >75% |

**实验脚本**:
```python
for attack_rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
    num_malicious = int(10 * attack_rate)
    num_benign = 10 - num_malicious
    config.num_benign = num_benign
    run_experiment(config)
```

---

### 2.4 客户端规模实验 (Exp-Scale)

**配置**: 标签翻转, 30%攻击率

| 客户端数 | 恶意数 | 配置 | 预期TPR | 预期精度 |
|---------|--------|------|---------|---------|
| 10 | 3 | num_clients=10 | >90% | >85% |
| 20 | 6 | num_clients=20 | >88% | >83% |
| 50 | 15 | num_clients=50 | >85% | >80% |
| 100 | 30 | num_clients=100 | >80% | >75% |

**注意**: 
- 增加客户端数时, 需要调整buffer_capacity
- 建议: buffer_capacity = num_clients * 50

---

### 2.5 数据分布实验 (Exp-NonIID)

#### 2.5.1 IID分布 (当前)

```python
def build_iid_dataloaders(dataset, num_clients):
    indices = np.random.permutation(len(dataset))
    for i in range(num_clients):
        start = i * len(dataset) // num_clients
        end = (i+1) * len(dataset) // num_clients
        yield dataset[indices[start:end]]
```

#### 2.5.2 Non-IID分布 (Dirichlet)

```python
from numpy.random import dirichlet

def build_noniid_dataloaders(dataset, num_clients, alpha=0.5):
    num_classes = 10
    for client_id in range(num_clients):
        # Dirichlet分布: α越小越non-IID
        label_dist = dirichlet([alpha] * num_classes)
        # 按分布采样
        client_data = []
        for label in range(num_classes):
            label_indices = np.where(dataset.targets == label)[0]
            num_samples = int(len(dataset) / num_clients * label_dist[label])
            sampled = np.random.choice(label_indices, num_samples, replace=False)
            client_data.extend(sampled)
        yield dataset[client_data]
```

**配置**:
```python
# IID
alpha = float('inf')  # 完全IID

# Non-IID (轻度)
alpha = 1.0

# Non-IID (中度)
alpha = 0.5

# Non-IID (重度)
alpha = 0.1
```

**实验**:
```python
for alpha in [float('inf'), 1.0, 0.5, 0.1]:
    config.data_alpha = alpha
    run_experiment(config)
```

**预期**:
- IID: TPR >90%, 精度 >85%
- Non-IID (α=0.5): TPR >85%, 精度 >80%
- Non-IID (α=0.1): TPR >75%, 精度 >70%

---

### 2.6 防御方法对比 (Exp-Defense)

#### 2.6.1 FedAvg (无防御)

```python
def fedavg_aggregation(client_updates):
    K = len(client_updates)
    return sum(client_updates) / K
```

#### 2.6.2 Trimmed Mean

```python
def trimmed_mean_aggregation(client_updates, trim_ratio=0.2):
    K = len(client_updates)
    trim_count = int(K * trim_ratio)
    
    # 按范数排序
    norms = [torch.norm(u) for u in client_updates]
    sorted_idx = torch.argsort(torch.tensor(norms))
    
    # 去掉最大和最小的
    kept_idx = sorted_idx[trim_count:-trim_count]
    kept_updates = [client_updates[i] for i in kept_idx]
    
    return sum(kept_updates) / len(kept_updates)
```

#### 2.6.3 Krum

```python
def krum_aggregation(client_updates, m=None):
    K = len(client_updates)
    if m is None:
        m = K - 2  # 选择最近的K-2个
    
    # 计算距离矩阵
    distances = torch.zeros(K, K)
    for i in range(K):
        for j in range(K):
            distances[i, j] = torch.norm(client_updates[i] - client_updates[j])
    
    # 选择距离最小的m个
    scores = torch.sum(torch.topk(distances, m, dim=1)[0], dim=1)
    best_idx = torch.argmin(scores)
    
    return client_updates[best_idx]
```

#### 2.6.4 AE-SVDD (你的方法)

```python
# 使用现有实现
server.phase1_step() / server.phase2_step()
```

#### 2.6.5 FLARE (如可实现)

```python
def flare_aggregation(client_updates, reputation_scores, threshold=0.5):
    # 多维声誉评分
    # 软权重聚合
    weights = torch.exp(-reputation_scores / temperature)
    weights = weights / weights.sum()
    
    aggregated = sum(w * u for w, u in zip(weights, client_updates))
    return aggregated
```

**对比表**:
```python
methods = {
    'FedAvg': fedavg_aggregation,
    'Trimmed Mean': trimmed_mean_aggregation,
    'Krum': krum_aggregation,
    'AE-SVDD': ae_svdd_aggregation,
}

for method_name, method_fn in methods.items():
    results[method_name] = run_experiment(config, aggregation_fn=method_fn)
```

---

## 3. 评估指标

### 3.1 检测性能指标

```python
# 混淆矩阵
TP = 恶意客户端被正确检测
FP = 良性客户端被误判为恶意
TN = 良性客户端被正确识别
FN = 恶意客户端未被检测

# 指标
TPR = TP / (TP + FN)           # 恶意检测率 (越高越好)
FPR = FP / (FP + TN)           # 良性误判率 (越低越好)
Precision = TP / (TP + FP)     # 检测准确率
Recall = TPR                    # 召回率
F1 = 2 * Precision * Recall / (Precision + Recall)
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### 3.2 模型性能指标

```python
# 全局模型精度
test_accuracy = 正确预测数 / 总样本数

# 收敛速度
convergence_rounds = 达到90%精度所需轮数
convergence_speed = 无攻击收敛轮数 / 有防御收敛轮数

# 鲁棒性提升
robustness_gain = (防御精度 - 无防御精度) / 无攻击精度 * 100%
```

### 3.3 计算效率指标

```python
# 时间成本
detection_time = 检测恶意客户端的时间
aggregation_time = 聚合的时间
total_time = 每轮总时间

# 内存成本
buffer_memory = 重放缓冲占用内存
model_memory = 模型占用内存
total_memory = 总内存占用
```

### 3.4 隐私指标

```python
# BN特征泄露风险
feature_dim = BN特征维度 (4480)
compressed_dim = 压缩后维度 (64)
compression_ratio = feature_dim / compressed_dim

# 差分隐私成本 (如实现)
epsilon = 隐私预算
delta = 失败概率
```

---

## 4. 实验流程

### 4.1 Phase 1: 基础验证 (Week 1)

**目标**: 验证AE-SVDD在基础配置下的有效性

```
Exp-Base (标签翻转, 10客户端, 30%攻击率, IID)
├─ 无防御 (FedAvg)
├─ Trimmed Mean
├─ Krum
└─ AE-SVDD

输出: 
- 精度曲线
- TPR/FPR
- 收敛速度对比
```

**检查点**:
- [ ] AE-SVDD TPR > 85%
- [ ] AE-SVDD 精度 > 80%
- [ ] 优于Trimmed Mean和Krum

### 4.2 Phase 2: 攻击鲁棒性 (Week 2)

**目标**: 评估对多种攻击的防御能力

```
Exp-Attack (10客户端, 30%攻击率, IID)
├─ 标签翻转
├─ 高斯噪声
├─ 符号翻转
├─ ALIE
└─ Statistical Mimicry

输出:
- 每种攻击的TPR/FPR
- 攻击难度排序
- 防御覆盖率
```

**检查点**:
- [ ] 标签翻转 TPR > 90%
- [ ] 高斯噪声 TPR > 85%
- [ ] Statistical Mimicry TPR > 70%

### 4.3 Phase 3: 可扩展性 (Week 3)

**目标**: 评估不同规模下的性能

```
Exp-Rate (标签翻转, 10客户端, IID)
├─ 10% 攻击率
├─ 20% 攻击率
├─ 30% 攻击率
├─ 40% 攻击率
└─ 50% 攻击率

Exp-Scale (标签翻转, 30%攻击率, IID)
├─ 10 客户端
├─ 20 客户端
├─ 50 客户端
└─ 100 客户端

输出:
- 攻击率 vs TPR/精度
- 客户端数 vs TPR/精度
- 可扩展性分析
```

**检查点**:
- [ ] 50%攻击率 TPR > 75%
- [ ] 100客户端 TPR > 75%

### 4.4 Phase 4: 非IID数据 (Week 4)

**目标**: 评估在异构数据下的性能

```
Exp-NonIID (标签翻转, 10客户端, 30%攻击率)
├─ IID (α=∞)
├─ Non-IID 轻度 (α=1.0)
├─ Non-IID 中度 (α=0.5)
└─ Non-IID 重度 (α=0.1)

输出:
- 数据异构性 vs TPR/精度
- 性能下降分析
```

**检查点**:
- [ ] Non-IID (α=0.5) TPR > 80%
- [ ] 性能下降 < 10%

### 4.5 Phase 5: 方法对比 (Week 5)

**目标**: 与最新方法对标

```
Exp-Defense (标签翻转, 100客户端, 30%攻击率, Non-IID)
├─ FedAvg
├─ Trimmed Mean
├─ Krum
├─ AE-SVDD
└─ FLARE (如可实现)

输出:
- 方法对比表
- 性能排序
- 优劣分析
```

**检查点**:
- [ ] AE-SVDD 优于 Trimmed Mean/Krum
- [ ] 接近或超过 FLARE

---

## 5. 实验代码框架

### 5.1 主实验脚本

```python
# experiments/run_all.py

import json
from pathlib import Path
from config import FedConfig
from main import run_federated

RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)

def run_experiment(config, exp_name):
    """运行单个实验"""
    print(f"Running {exp_name}...")
    
    results = run_federated(config, use_svdd=True)
    
    # 保存结果
    result_file = RESULTS_DIR / f"{exp_name}.json"
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    # Phase 1: 基础验证
    config = FedConfig()
    run_experiment(config, "exp_base_aesvdd")
    
    # Phase 2: 攻击类型
    for attack in ["label_flipping", "gaussian_noise", "sign_flipping"]:
        config.attack_type = attack
        run_experiment(config, f"exp_attack_{attack}")
    
    # Phase 3: 攻击率
    for rate in [0.1, 0.2, 0.3, 0.4, 0.5]:
        config.num_benign = int(10 * (1 - rate))
        run_experiment(config, f"exp_rate_{int(rate*100)}")
    
    # Phase 4: 客户端数
    for num_clients in [10, 20, 50, 100]:
        config.num_clients = num_clients
        config.num_benign = int(num_clients * 0.7)
        run_experiment(config, f"exp_scale_{num_clients}")
    
    # Phase 5: 方法对比
    for method in ["fedavg", "trimmed_mean", "krum", "aesvdd"]:
        run_experiment(config, f"exp_defense_{method}")

if __name__ == "__main__":
    main()
```

### 5.2 结果分析脚本

```python
# experiments/analyze_results.py

import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("./results")

def load_results(exp_name):
    with open(RESULTS_DIR / f"{exp_name}.json") as f:
        return json.load(f)

def plot_accuracy_curves():
    """绘制精度曲线"""
    results = load_results("exp_base_aesvdd")
    
    plt.figure(figsize=(10, 6))
    plt.plot(results['accuracy'], label='AE-SVDD')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(RESULTS_DIR / "accuracy_curve.png")

def plot_tpr_fpr():
    """绘制TPR/FPR对比"""
    methods = ["fedavg", "trimmed_mean", "krum", "aesvdd"]
    tprs = []
    fprs = []
    
    for method in methods:
        results = load_results(f"exp_defense_{method}")
        tprs.append(results['tpr'])
        fprs.append(results['fpr'])
    
    plt.figure(figsize=(10, 6))
    plt.scatter(fprs, tprs)
    for i, method in enumerate(methods):
        plt.annotate(method, (fprs[i], tprs[i]))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig(RESULTS_DIR / "roc_curve.png")

def generate_summary_table():
    """生成总结表"""
    data = []
    
    for exp_name in RESULTS_DIR.glob("*.json"):
        results = load_results(exp_name.stem)
        data.append({
            'Experiment': exp_name.stem,
            'Accuracy': results.get('accuracy', 0),
            'TPR': results.get('tpr', 0),
            'FPR': results.get('fpr', 0),
            'F1': results.get('f1', 0),
        })
    
    df = pd.DataFrame(data)
    df.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(df)

if __name__ == "__main__":
    plot_accuracy_curves()
    plot_tpr_fpr()
    generate_summary_table()
```

---

## 6. 预期结果与基准

### 6.1 基础配置预期

| 指标 | 无防御 | Trimmed Mean | Krum | AE-SVDD |
|------|--------|-------------|------|---------|
| 精度 | 15% | 70% | 75% | 85% |
| TPR | - | 75% | 80% | 90% |
| FPR | - | 15% | 10% | 5% |
| F1 | - | 0.80 | 0.85 | 0.92 |

### 6.2 攻击类型预期

| 攻击类型 | 无防御 | AE-SVDD TPR | AE-SVDD 精度 |
|---------|--------|------------|------------|
| 标签翻转 | 10% | 92% | 87% |
| 高斯噪声 | 20% | 88% | 82% |
| 符号翻转 | 35% | 85% | 78% |
| ALIE | 45% | 78% | 72% |
| Statistical Mimicry | 40% | 72% | 68% |

### 6.3 攻击率预期

| 攻击率 | 无防御精度 | AE-SVDD TPR | AE-SVDD 精度 |
|--------|-----------|------------|------------|
| 10% | 50% | 96% | 91% |
| 20% | 30% | 93% | 88% |
| 30% | 15% | 90% | 85% |
| 40% | 10% | 85% | 80% |
| 50% | 5% | 80% | 75% |

---

## 7. 时间表与里程碑

| 周 | 任务 | 交付物 | 检查点 |
|----|------|--------|--------|
| W1 | Phase 1: 基础验证 | 精度曲线, TPR/FPR | AE-SVDD TPR>85% |
| W2 | Phase 2: 攻击鲁棒性 | 攻击对比表 | 5种攻击覆盖 |
| W3 | Phase 3: 可扩展性 | 规模分析图 | 100客户端可行 |
| W4 | Phase 4: 非IID数据 | 异构性分析 | Non-IID性能>80% |
| W5 | Phase 5: 方法对比 | 对标表格 | 优于基线方法 |
| W6 | 论文撰写 | 论文初稿 | 完整实验报告 |

---

## 8. 快速参考

### 运行单个实验

```bash
# 基础配置
python new/main.py --config new/config.py --use_svdd True

# 标签翻转
python new/main.py --attack_type label_flipping --num_benign 7

# 100客户端
python new/main.py --num_clients 100 --num_benign 70

# Non-IID数据
python new/main.py --data_alpha 0.5
```

### 批量运行

```bash
cd experiments
python run_all.py
python analyze_results.py
```

---

**文档版本**: v1.0
**最后更新**: 2026-03-25
