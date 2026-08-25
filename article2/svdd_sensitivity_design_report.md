# AE-SVDD 敏感性分析实验设计报告

## 1. 实验目标

本实验分成两个彼此独立的部分：

1. **AE-SVDD 参数敏感性分析**：研究 `lambda`、P1 时长、Trust size 和 latent dimension 对检测效果和全局模型效果的影响。
2. **攻击强度鲁棒性分析**：固定 AE-SVDD 配置，只改变恶意参与方比例，研究防御在不同攻击规模下的退化情况。

恶意参与方比例不再作为 AE-SVDD 参数敏感性中的变量；它只出现在单独的鲁棒性分析中。

## 2. 固定实验协议

| 项目 | 设置 |
|---|---|
| 数据集 | MNIST、FashionMNIST、CIFAR10、AGNews |
| 客户端总数 | 100 |
| 默认恶意客户端数 | 30，即恶意比例 0.3 |
| 参与方式 | 每轮全部客户端参与 |
| 通信轮数 | 300 |
| 本地 epoch | 1 |
| batch size | 64 |
| 数据划分 | Dirichlet alpha = 1.0 |
| 输入模式 | `absolute` |
| descriptor | 4096 维固定 descriptor |
| 标准化 | median/MAD |
| Top-K 候选比例 | 10%、20%、30%、40% |
| 随机种子 | 42、43、44 |
| 防御 | AE-SVDD |

AGNews 不运行 BD，因为当前数据和攻击实现不支持图像触发器。其余数据集运行 GN、SF、LF、BD、LIE。

因此任务-攻击组合数为：

```text
MNIST/FashionMNIST/CIFAR10: 3 × 5 = 15
AGNews:                      1 × 4 = 4
总计:                         19 个任务-攻击组合
```

## 3. AE-SVDD 默认配置

除正在扫描的参数外，其余 AE-SVDD 配置固定如下：

| 参数 | 默认值 |
|---|---:|
| `svdd_lambda` | 0.5 |
| `phase1_rounds` | 15 |
| `server_validation_size` | 50 |
| `latent_dim` | 64 |
| Phase 1 score | `recon` |
| Phase 2 score | `combined` |
| `svdd_input_dim` | 4096 |
| `svdd_input_mode` | `absolute` |
| `svdd_normalization` | `median_mad` |

`lambda` 的定义为：

```text
total_loss = lambda × svdd_loss + (1 - lambda) × reconstruction_loss
```

它不是恶意比例，也不是 Dirichlet alpha。

## 4. 参数敏感性分析

采用**单因素变化**设计：每次只改变一个参数，其他参数使用第 3 节的默认值。不采用全笛卡尔积，避免把参数交互效应混入单因素结论。

### 4.1 lambda 敏感性

扫描：

```text
svdd_lambda ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
```

固定：

```text
恶意比例 = 0.3
P1 = 15
Trust size = 50
latent = 64
```

这部分回答：AE 重构损失和 SVDD 紧致性损失的权重如何影响防御效果。

### 4.2 P1 时长敏感性

扫描：

```text
phase1_rounds ∈ {5, 15, 30, 50, 100}
```

固定：

```text
lambda = 0.5
恶意比例 = 0.3
Trust size = 50
latent = 64
```

P1 仍使用 `recon` score，P2 仍使用 `combined` score。总轮数始终是 300。

### 4.3 Trust size 敏感性

扫描：

```text
server_validation_size ∈ {10, 25, 50, 100, 200}
```

这里的 Trust size 指服务器持有的干净验证样本数，不是客户端数量。

固定：

```text
lambda = 0.5
恶意比例 = 0.3
P1 = 15
latent = 64
```

### 4.4 latent dimension 敏感性

扫描：

```text
latent_dim ∈ {16, 32, 64, 128}
```

固定：

```text
lambda = 0.5
恶意比例 = 0.3
P1 = 15
Trust size = 50
```

输入 descriptor 始终保持 4096 维，只改变 AE 的 latent bottleneck 维度。

### 4.5 参数敏感性实验数量

每个任务-攻击-种子包含：

```text
lambda:       7 组
P1:           5 组
Trust size:   5 组
latent:       4 组
合计:         21 组
```

总任务数：

```text
19 个任务-攻击组合 × 21 个参数配置 × 3 个种子 = 1197 个实验
```

## 5. 恶意比例鲁棒性分析

这部分不用于分析 AE-SVDD 参数，而用于分析攻击规模变化时的防御退化。

固定：

```text
lambda = 0.5
P1 = 15
Trust size = 50
latent = 64
```

扫描恶意比例：

```text
{0.2, 0.3, 0.4}
```

每个比例下：

```text
num_clients = 100
num_malicious = 100 × ratio
```

本报告将恶意比例限制在不超过 0.4，因此 LIE 使用当前论文公式，不需要额外的高比例特殊处理。

鲁棒性分析的实验数为：

```text
19 个任务-攻击组合 × 3 个比例 × 3 个种子 = 171 个实验
```

## 6. 攻击配置

| 攻击 | 适用数据集 | 说明 |
|---|---|---|
| GN | 全部数据集 | Gaussian noise Byzantine attack |
| SF | 全部数据集 | Sign-flipping attack |
| LF | 全部数据集 | Label-flipping attack |
| BD | MNIST、FashionMNIST、CIFAR10 | 图像后门攻击，AGNews 为 N/A |
| LIE | 全部数据集 | 在恶意比例 0.2、0.3、0.4 下运行 |

恶意客户端身份只用于计算检测指标，防御过程不得读取真实恶意标签。

## 7. 统计指标

每个配置运行 3 个种子，并保留每轮结果。

### 7.1 检测指标

- **DAR**：所有客户端的正确分类率。
- **DPR**：被拒绝客户端中真正恶意客户端的比例。
- **RR/TPR**：恶意客户端被正确拒绝的比例。
- **FPR**：被误拒绝的 benign 客户端比例。
- **selected reject ratio**：最终选择的 Top-K 拒绝比例。

### 7.2 全局模型指标

- **ACC/TACC**：干净测试集准确率。
- **ASR**：仅对 BD 报告；非 BD 攻击记为 N/A。

### 7.3 汇总方法

主表使用最后 10 轮的均值：

```text
每个 seed：最后 10 轮均值
最终结果：3 个 seed 的均值 ± 标准差
```

不使用单个偶然轮次作为最终结论。对于 BD，DAR、ACC、ASR 必须同时报告；不能只报告 DAR。

## 8. 结果文件和完整性检查

每个结果文件必须满足：

1. 任务、攻击、seed 与目录配置一致；
2. `num_malicious` 与该鲁棒性比例一致，或在参数敏感性中固定为 30；
3. `svdd_lambda` 与当前敏感性变量一致；
4. `phase1_rounds`、`server_validation_size`、`latent_dim` 与目录标签一致；
5. `svdd_input_dim=4096`；
6. `svdd_input_mode=absolute`；
7. `svdd_normalization=median_mad`；
8. 结果包含完整 300 轮；
9. 结果没有 NaN/Inf；
失败任务必须单独记录原因，不能直接从分母中删除而不报告。

## 9. 当前已有结果的处理建议

之前错误 runner 产生的结果可以拆分处理：

- `lambda=0.5、P1=15、Trust=50、latent=64` 的恶意比例变化结果，可以作为鲁棒性分析的已有结果。
- `恶意比例=0.3、lambda=0.5` 的 P1、Trust、latent 结果，可以作为对应参数敏感性的部分已有结果。
- 恶意比例不为 0.3 的结果不能放入 lambda/P1/Trust/latent 参数敏感性表，但可以放入恶意比例鲁棒性表。
- 旧的 absolute/delta 对比结果只作为独立对比实验，不与本报告的 absolute + median/MAD 主矩阵混合汇总。

## 10. 执行前需要确认的事项

开始训练前只需要确认以下选择：

1. 是否采用本报告的**单因素敏感性设计**，而不是 700 组全笛卡尔积设计；
2. 是否复用之前已完成且配置完全匹配的结果；
3. Top-K 是否继续使用内部候选 `{10%, 20%, 30%, 40%}`，并选择验证集准确率最高者作为最终聚合比例。

确认后再生成最终 runner 和实验 manifest，不提前启动训练。
