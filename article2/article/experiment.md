# AE-SVDD 联邦学习恶意参与方检测：当前实验协议

**更新日期：2026-09-15**
**代码基线：当前工作区 `src/` 与 `configs/`**

本文档描述当前仓库中可以直接执行的实验协议。它以代码为准，不把尚未注册的攻击、防御方法或旧脚本当作可复现实验的一部分。方法细节见根目录的 `method.md`。

## 1. 执行入口与配置优先级

实验从模块化入口启动：

```bash
dl/bin/python -m src.pipeline --config <pipeline-config.json>
```

管线配置包含 `task`、`attacks`、`defenses`、`log_dir`、基础联邦配置文件、超参数表和可选的 `fed_config_overrides`。逗号分隔的任务、攻击或防御会展开为组合实验；`all` 会展开该类别的全部已注册项。

单次运行的有效配置按如下顺序生成：

1. `configs/federated.json` 的 `values`；
2. 管线 JSON 的 `fed_config_overrides`；
3. `configs/hyperparameters.json` 中匹配的公共、攻击、任务和防御配置；
4. 管线 JSON 的 `fed_config_overrides` 再次覆盖，以便实验矩阵明确固定变量。

运行结果写入 `log_dir`，其中的结构化 JSON 会记录任务、攻击、防御、随机种子、有效配置、逐轮指标和最终测试结果。复现实验时应使用该有效配置，而不是只引用配置文件的默认值。

## 2. 数据集与模型

当前 `TASK_REGISTRY` 中的任务如下。服务器先从训练集确定性地抽取按类别均衡的干净验证样本，并将其从所有客户端训练数据中移除；随后才进行客户端划分。

| 任务 ID | 数据集 | 类别数 | 全局模型 |
| --- | --- | ---: | --- |
| `mnist` | MNIST | 10 | 灰度 LeNet 分类器 |
| `fashion_mnist` | Fashion-MNIST | 10 | Fashion-MNIST CNN |
| `cifar10` | CIFAR-10 | 10 | CIFAR-10 适配的 ResNet-18 |
| `covid19` | COVID-19 Radiography Database | 4 | ImageNet 预训练 ResNet-50 |
| `ag_news` | AG News | 4 | 轻量 Transformer 文本分类器 |

`dirichlet_alpha=null` 选择 IID 客户端划分；正数选择严格 Dirichlet 划分，数值越小代表数据异构性越强。`dirichlet_noniid_beta` 仅用于兼容旧配置：当 `dirichlet_alpha` 为 `null` 时才会读取它。

## 3. 默认联邦学习设置

除非管线覆盖，`configs/federated.json` 的默认设置为：

| 设置 | 默认值 |
| --- | --- |
| 客户端数 / 恶意客户端数 | 100 / 30 |
| 通信轮数 | 300 |
| 每轮参与 | 全部客户端 |
| 本地训练 | 1 epoch，batch size 64 |
| 服务器干净验证样本 | 50 个，类别均衡且不分发给客户端 |
| 初始学习率 / momentum / 权重衰减 | 0.05 / 0.9 / 5e-4（任务配置可覆盖） |
| 默认数据划分 | Dirichlet alpha = 1.0 |
| 随机种子 / 设备 | 42 / `cuda` |

任务专属的学习率和权重衰减分别由 `configs/hyperparameters.json` 固定：MNIST 为 `0.1/1e-4`，Fashion-MNIST 为 `0.1/0`，CIFAR-10 为 `0.05/1e-4`，COVID-19 为 `0.005/5e-4`，AG News 为 `0.1/0`。COVID-19 任务还默认使用均衡客户端采样。

## 4. 已实现攻击

只能使用下表中 `ATTACK_REGISTRY` 已注册的攻击 ID。`gaussian_noise`、`label_flipping`、`sign_flipping`、`alie` 等长名称会分别规范化为 `gn`、`lf`、`sf`、`lie`。

| ID | 类型 | 当前实现 | 关键默认参数 |
| --- | --- | --- | --- |
| `none` | 对照 | 正常本地训练；管线会将恶意客户端数置为 0 | — |
| `lf` | 数据投毒 | 对称标签映射 `y' = C - 1 - y` | 由任务类别数决定 |
| `gn` | 模型投毒 | 以全局模型每个浮点张量的均值和标准差生成高斯替换上传 | `gaussian_sigma=0.3` |
| `sf` | 模型投毒 | 上传 `W_g - s(W_l-W_g)` | `sign_flip_scale=1.0` |
| `lie` | 协同模型投毒 | ALIE/LIE 在模型增量空间构造 `mu + z sigma` | `lie_z_override=null`，由客户端数量推导 |
| `minmax` | 协同模型投毒 | Min-Max 距离约束攻击 | `distance_attack_deviation=std` |
| `minsum` | 协同模型投毒 | Min-Sum 距离约束攻击 | `distance_attack_deviation=std` |
| `bd` | 后门 | 图像触发器、目标标签和模型替换 | target=0，poison=0.6，trigger=5，value=1.0，scale=3.0 |
| `lit` | 后门 | FedDMC LIT 变体 | 使用对应 FedDMC 参数 |
| `scaling` | 后门 | FedDMC Scaling 变体 | 使用对应 FedDMC 参数 |
| `mix` | 混合攻击 | 将多个已注册攻击分配给不同恶意客户端 | `mixed_attack_types` 指定集合 |

仓库没有注册 Statistical Mimicry、FLARE、GShield 或 FedTrident；它们不属于本实验协议。AG News 不运行需要图像触发器的后门攻击。

## 5. 已实现防御

所有基线均从同一批客户端上传状态开始。当前 `DEFENSE_REGISTRY` 提供：

| ID | 方法 |
| --- | --- |
| `avg` | FedAvg |
| `tm` | Trimmed Mean |
| `mk` | Multi-Krum |
| `svdd` | AE-SVDD（本文方法） |
| `dmc` | FedDMC-style 多视图检测 |
| `lasa` | LASA |
| `seca` | FedSECA |
| `fld` | FL-Defender |
| `alignins` | AlignIns |
| `bnguard` | BNGuard |
| `flgmm` | FLGMM |
| `flanders` | FLANDERS |

比较时，所有方法必须使用相同的任务、种子、客户端集合、攻击、通信轮数和划分参数。仅当这些条件一致时，才可横向比较最终精度、检测结果或运行开销。

## 6. AE-SVDD 协议

### 6.1 输入、评分和筛选

AE-SVDD 从每个客户端的全部可训练模型参数生成固定的 4096 维分层 CountSketch 描述符。实现以零状态为参考，因此输入表示为客户端的绝对参数状态；描述符在当轮客户端之间按逐特征 median/MAD 标准化。无穷或 NaN 描述符行不会被接收。

默认自编码器为 `4096 → 256 → 64 → 256 → 4096`，编码端包含 LeakyReLU 和 LayerNorm，重建误差为逐客户端平均绝对误差。Phase 1 的前 15 轮仅按重建误差评分；最后一轮筛选后使用被接收客户端的嵌入初始化 SVDD 中心。此后 Phase 2 将重建误差和到中心的平方距离分别做 median/MAD 标准化并相加。

客户端选择使用硬 MAD 阈值：

```text
accepted = finite(score) and score <= median(score) + svdd_mad_k * MAD(score)
```

默认 `svdd_mad_k=0.5`。被拒绝客户端的聚合权重为 0，保留客户端等权聚合；该选择过程不使用验证标签。服务器仍会创建干净验证集，以支持其它防御或实验记录，但当前 AE-SVDD 的 `mad_threshold` 路径不会用它来选择客户端。

### 6.2 优化设置

| 设置 | 默认值 |
| --- | --- |
| 描述符维度 / 种子 | 4096 / 2027 |
| latent dimension | 64 |
| AE 学习率 / 权重衰减 / 梯度裁剪 | 1e-3 / 1e-6 / 1.0 |
| Phase 1 轮数 | 15 |
| 中心 EMA 系数 | 0.9 |
| Phase 2 损失 | `0.5 * SVDD + 0.5 * reconstruction` |
| Phase 2 重建分位数 | 0.8 |
| SVDD 梯度裁剪 | 1.0 |

在 Phase 2 中，SVDD 分支会冻结解码器；重建分支只在已接收客户端内取低重建误差分位数训练。筛选、中心更新和模型更新都不会让当轮被拒绝客户端影响 AE/SVDD 状态或最终聚合。

`svdd_input_mode` 和 `svdd_normalization` 是兼容配置字段；当前 `SVDDDefense` 固定使用 absolute 输入和 median/MAD 归一化。实验结果应报告实际代码路径，而不是将这些兼容字段误解为可切换的算法变体。

## 7. 推荐实验矩阵

### 7.1 主对比

对每个任务选择适用攻击，比较 `avg`、`tm`、`mk`、`svdd` 和必要的已注册扩展基线。固定 100 个客户端、30 个恶意客户端、300 轮、相同种子和相同划分。报告干净测试准确率；图像后门任务同时报告 ASR。客户端身份只用于离线计算检测指标，绝不能传入防御器。

### 7.2 参数敏感性

`tools/run_svdd_52_sensitivity.py` 支持只改变一个 AE-SVDD 因子的加性扫描，而非笛卡尔积。可扫描的因素包括 `svdd_lambda`、`phase1_rounds`、`server_validation_size` 和 `latent_dim`；应固定任务、攻击、数据划分、客户端组成和其余 AE-SVDD 设置。默认扫描使用 MNIST 与 Fashion-MNIST、`gn`、`svdd`、300 轮和种子 42；命令行参数可改变这些值。

### 7.3 鲁棒性

`tools/run_svdd_robustness.py` 以恶意比例（10%、20%、30%、40%）和 Dirichlet alpha（5.0、1.0、0.5、0.1）为单因素变量。报告时需将恶意比例与数据异构性分开解释：低 alpha 造成的良性客户端漂移不能被直接视作攻击检出。

## 8. 评估与报告要求

每个结论必须来自完整的结构化结果文件，且结果元数据应与计划条件一致。至少报告：

- 最终测试 accuracy；类别不平衡任务同时报告 balanced accuracy；
- 图像后门和混合后门条件的 ASR；
- 客户端接收掩码、选择分数、保留比例和最终聚合权重；
- 恶意客户端身份可见的离线检测统计（例如召回率、误报率、精确率和 F1）；
- 完整的有效配置、随机种子、轮数、失败情况和运行环境。

不要将不同任务的原始 accuracy 直接平均；跨任务分析应以同一任务内的配对比较、相对变化和检测统计为依据。尚未完成或不可解析的运行应显式标为缺失，不能用预期数值替代。

## 9. 快速检查

在启动训练前，可先只验证组合是否合法：

```bash
dl/bin/python -m src.pipeline --config configs/pipeline_smoke.json --dry-run
```

该命令只展开并校验任务、攻击和防御，不加载数据或执行训练。正式实验应使用独立的管线 JSON，并将该 JSON 与结果文件一同保存。
