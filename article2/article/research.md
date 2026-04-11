# 无根数据（Data-Free）联邦学习领域调研报告

> 调研时间：2024-2025年相关论文  
> 生成日期：2026年3月24日

---

## 目录

1. [概述](#概述)
2. [核心方法分类](#核心方法分类)
3. [重点论文详解](#重点论文详解)
4. [实验设置对比](#实验设置对比)
5. [开源代码汇总](#开源代码汇总)
6. [发展趋势与挑战](#发展趋势与挑战)
7. [参考文献](#参考文献)

---

## 概述

无根数据联邦学习（Data-Free Federated Learning）是指在服务器端**不访问任何原始数据**的情况下，通过知识蒸馏、生成模型等技术聚合来自分布式客户端的模型知识。这一方向在隐私保护和通信效率方面具有重要意义。

### 主要应用场景

- **医疗数据协作**：医院间共享模型而不共享患者数据
- **跨设备学习**：移动设备、IoT设备的联邦训练
- **隐私敏感领域**：金融、个人数据等无法集中存储的场景

---

## 核心方法分类

### 1. 基于生成模型的知识蒸馏

利用生成对抗网络（GAN）或扩散模型合成伪数据，用于服务器端的知识蒸馏。

**代表方法**：
- 双生成器对抗蒸馏（DFDG）
- 三玩家生成对抗网络（FedDTG）
- 扩散模型驱动的数据重放

### 2. 单次通信联邦学习（One-Shot FL）

仅通过一轮通信完成全局模型训练，极大降低通信开销。

**代表方法**：
- 合成蒸馏器-馏出物通信（FedSD2C）
- 高斯头OFL家族
- 数据自由双生成器对抗蒸馏（DFDG）

### 3. 异步联邦学习中的数据自由方法

解决异步联邦学习中过时更新（stale updates）问题。

**代表方法**：
- FedRevive：通过数据自由知识蒸馏恢复过时更新

### 4. 异构联邦学习

处理客户端模型架构不同或数据分布高度异构的场景。

**代表方法**：
- FedBiCross：双层优化框架
- HFedCKD：双向对比学习
- Mosaic：混合专家模型

### 5. 持续学习与灾难遗忘缓解

在联邦学习中处理连续任务而不遗忘先前知识。

**代表方法**：
- FedDCL：数据自由持续学习
- 扩散驱动的数据重放

---

## 重点论文详解

### 1. FedRevive: 异步联邦学习中的数据自由知识蒸馏

**论文信息**：
- 标题：Reviving Stale Updates: Data-Free Knowledge Distillation for Asynchronous Federated Learning
- 作者：Baris Askin, Holger R. Roth, Zhenyu Sun, et al.
- 发表：arXiv:2511.00655, 2025
- 链接：https://arxiv.org/abs/2511.00655

**核心问题**：
异步联邦学习（AFL）允许客户端独立通信，但会引入基于过时全局模型计算的更新（staleness），导致优化不稳定和收敛困难。

**方法概述**：
1. **参数空间聚合**：整合来自客户端的模型更新
2. **服务器端DFKD**：无需数据访问，将过时客户端更新的知识转移到当前全局模型
3. **元学习生成器**：合成用于多教师蒸馏的伪样本
4. **混合聚合方案**：结合原始更新与DFKD更新，缓解过时问题同时保持AFL可扩展性

**实验设置**：
- **数据集**：多个视觉和文本基准数据集
- **客户端数量**：未明确说明
- **异构设置**：IID和非IID数据分布
- **评估指标**：
  - 训练速度提升：最高38.4%
  - 最终准确率提升：最高16.5%

**创新点**：
- 首次将数据自由知识蒸馏应用于异步联邦学习
- 提出混合聚合机制平衡原始更新和蒸馏更新
- 使用元学习生成器提高合成数据质量

---

### 2. FedBiCross: 医疗数据上的非IID数据自由单次联邦学习

**论文信息**：
- 标题：FedBiCross: A Bi-Level Optimization Framework to Tackle Non-IID Challenges in Data-Free One-Shot Federated Learning on Medical Data
- 作者：Yuexuan Xia, Yinghao Zhang, Yalin Liu, et al.
- 发表：arXiv:2601.01901, 2026
- 链接：https://arxiv.org/abs/2601.01901

**核心问题**：
单次联邦学习（OSFL）在单次通信轮次中训练模型，但现有方法在非IID数据下表现不佳——冲突的预测在平均过程中相互抵消，产生接近均匀的软标签，为蒸馏提供弱监督。

**方法概述**：
FedBiCross提出三阶段个性化OSFL框架：

1. **客户端聚类**：按模型输出相似性聚类客户端，形成一致的子集成
2. **双层跨集群优化**：学习自适应权重，选择性利用有益的跨集群知识，同时抑制负迁移
3. **个性化蒸馏**：针对每个客户端特定适应

**实验设置**：
- **数据集**：四个医疗图像数据集
  - 可能包括：胸部X光、病理切片等（具体数据集名称论文未详细说明）
- **数据分布**：不同程度的非IID设置
- **评估指标**：
  - 在不同非IID程度下 consistently 超越SOTA基线

**创新点**：
- 针对医疗数据隐私需求设计
- 双层优化解决非IID挑战
- 个性化蒸馏适应不同客户端

---

### 3. FedSD2C: 合成蒸馏器-馏出物通信

**论文信息**：
- 标题：One-shot Federated Learning via Synthetic Distiller-Distillate Communication
- 作者：Junyuan Zhang, Songhua Liu, Xinchao Wang
- 发表：NeurIPS 2024, arXiv:2412.05186
- 链接：https://arxiv.org/abs/2412.05186
- **代码**：https://github.com/Carkham/FedSD2C

**核心问题**：
单次联邦学习面临两个主要挑战：
1. 数据异构性导致教师模型提供误导性知识
2. 复杂数据集上的可扩展性问题，存在两步信息损失（数据→模型→逆数据）

**方法概述**：
FedSD2C引入**蒸馏器**直接从本地数据合成信息丰富的**馏出物**：

1. **减少信息损失**：蒸馏器直接从原始数据学习，而非从本地模型间接学习
2. **解决数据异构性**：共享合成馏出物而非不一致的本地模型
3. **通信效率**：单次通信轮次完成训练

**实验设置**：
- **数据集**：更复杂和真实的数据集（具体名称未在摘要中说明）
- **基准对比**：其他单次联邦学习方法
- **评估指标**：
  - 相比最佳基线性能提升高达2.6倍

**创新点**：
- 提出"蒸馏器-馏出物"新范式
- 直接从数据合成而非模型反演
- 在复杂真实数据集上验证有效性

---

### 4. DFDG: 数据自由双生成器对抗蒸馏

**论文信息**：
- 标题：DFDG: Data-Free Dual-Generator Adversarial Distillation for One-Shot Federated Learning
- 作者：Kangyang Luo, Shuai Wang, Yexuan Fu, et al.
- 发表：ICDM 2024, arXiv:2409.07734
- 链接：https://arxiv.org/abs/2409.07734

**核心问题**：
单次联邦学习面临三个限制：
1. 需要公共数据集
2. 仅关注模型同质设置
3. 从本地模型蒸馏的知识有限

**方法概述**：
DFDG通过训练双生成器探索更广泛的本地模型训练空间：

1. **双生成器训练**：
   - 关注保真度（fidelity）、可迁移性（transferability）和多样性（diversity）
   - 定制交叉发散损失（cross-divergence loss）减少双生成器输出空间重叠

2. **双模型蒸馏**：
   - 训练好的双生成器协作提供全局模型更新的训练数据

**实验设置**：
- **数据集**：各种图像分类任务
- **对比基线**：SOTA单次联邦学习方法
- **评估指标**：
  - 准确率显著提升

**创新点**：
- 双生成器设计增加知识探索空间
- 对抗训练方式
- 无需公共数据集

---

### 5. FedDCL: 模型异构云-设备协作中的数据自由持续学习

**论文信息**：
- 标题：Data-Free Continual Learning of Server Models in Model-Heterogeneous Cloud-Device Collaboration
- 作者：Xiao Zhang, Zengzhe Chen, Yuan Yuan, et al.
- 发表：arXiv:2509.25977, 2025
- 链接：https://arxiv.org/abs/2509.25977

**核心问题**：
云-设备协作计算中，联邦学习面临：
1. 数据异构性
2. 模型异构性
3. 灾难性遗忘
4. 知识不对齐（新挑战）

**方法概述**：
FedDCL利用预训练扩散模型提取轻量级类别特定原型，实现三重数据自由优势：

1. **当前任务合成数据生成**：增强训练，对抗非IID数据分布
2. **无样本生成重放**：保留先前任务知识
3. **数据自由动态知识迁移**：从异构设备到云端

**实验设置**：
- **数据集**：各种数据集（具体名称未详细说明）
- **场景**：模型异构联邦设置
- **评估指标**：
  - 有效性和泛化能力验证

**创新点**：
- 结合扩散模型和持续学习
- 处理模型异构性
- 三重数据自由优势设计

---

### 6. HFedCKD: 基于数据自由知识蒸馏的鲁棒异构联邦学习

**论文信息**：
- 标题：HFedCKD: Toward Robust Heterogeneous Federated Learning via Data-free Knowledge Distillation and Two-way Contrast
- 作者：Yiting Zheng, Bohan Lin, Jinqian Chen, Jihua Zhu
- 发表：arXiv:2503.06511, 2025
- 链接：https://arxiv.org/abs/2503.06511

**核心问题**：
当前联邦学习方法难以有效处理异构性（数据和模型异构）。

**方法概述**：
- 数据自由知识蒸馏
- 双向对比学习（Two-way Contrast）

**实验设置**：
- 具体实验细节论文中未详细说明

---

### 7. Mosaic: 基于混合专家的数据自由知识蒸馏

**论文信息**：
- 标题：Mosaic: Data-Free Knowledge Distillation via Mixture-of-Experts for Heterogeneous Distributed Environments
- 作者：Junming Liu, Yanting Gao, Siyuan Meng, et al.
- 发表：arXiv:2505.19699, 2025
- 链接：https://arxiv.org/abs/2505.19699

**方法概述**：
- 使用混合专家模型（Mixture-of-Experts）进行数据自由知识蒸馏
- 针对异构分布式环境设计

**实验设置**：
- 43页论文，23个图，15个表

---

### 8. 扩散驱动的数据重放

**论文信息**：
- 标题：Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning
- 作者：Jinglin Liang, Jin Zhong, Hanlin Gu, et al.
- 发表：ECCV 2024 Oral, arXiv:2409.01128
- 链接：https://arxiv.org/abs/2409.01128

**核心问题**：
联邦类别持续学习中的灾难性遗忘问题。

**方法概述**：
- 利用扩散模型生成数据重放
- 无需存储真实历史数据

---

### 9. 数据自由联邦类别增量学习

**论文信息**：
- 标题：Data-Free Federated Class Incremental Learning with Diffusion-Based Generative Memory
- 作者：Naibo Wang, Yuchen Deng, Wenjie Feng, et al.
- 发表：arXiv:2405.17457, 2024
- 链接：https://arxiv.org/abs/2405.17457

**方法概述**：
- 基于扩散的生成记忆
- 数据自由联邦类别增量学习

---

### 10. FedTAD: 子图联邦学习的拓扑感知数据自由知识蒸馏

**论文信息**：
- 标题：FedTAD: Topology-aware Data-free Knowledge Distillation for Subgraph Federated Learning
- 作者：Yinlin Zhu, Xunkai Li, Zhengyu Wu, et al.
- 发表：IJCAI 2024, arXiv:2404.14061
- 链接：https://arxiv.org/abs/2404.14061

**应用场景**：
- 图神经网络（GNN）联邦学习
- 子图联邦学习场景

---

## 实验设置对比

| 论文 | 年份 | 数据集 | 客户端数 | 异构设置 | 主要评估指标 |
|------|------|--------|----------|----------|--------------|
| FedRevive | 2025 | 视觉+文本基准 | - | IID/非IID | 训练速度↑38.4%, 准确率↑16.5% |
| FedBiCross | 2026 | 4个医疗图像数据集 | - | 不同非IID程度 | 超越SOTA基线 |
| FedSD2C | 2024 | 复杂真实数据集 | - | 非IID | 性能提升2.6倍 |
| DFDG | 2024 | 图像分类任务 | - | 异构 | 准确率显著提升 |
| FedDCL | 2025 | 多个数据集 | - | 模型异构 | 泛化能力验证 |
| HFedCKD | 2025 | - | - | 异构 | - |
| Mosaic | 2025 | - | - | 异构 | - |
| 扩散驱动重放 | 2024 | - | - | 持续学习 | 遗忘缓解 |

### 常见数据集

**图像分类**：
- CIFAR-10/100
- MNIST/Fashion-MNIST
- ImageNet子集
- 医疗图像（胸部X光、病理切片等）

**文本数据**：
- 情感分析数据集
- 文本分类基准

**图数据**：
- Cora
- CiteSeer
- PubMed
- 分子图数据集

### 异构性设置

1. **数据异构性（Non-IID）**：
   - Dirichlet分布（α参数控制异构程度，常用0.1-0.5）
   - 按类别划分（每个客户端只有部分类别）
   - 按数量划分（不同客户端数据量差异）

2. **模型异构性**：
   - 不同架构（CNN、ResNet、ViT等）
   - 不同深度/宽度
   - 异构本地训练轮次

---

## 开源代码汇总

| 论文 | 代码链接 | 框架 | 备注 |
|------|----------|------|------|
| FedSD2C | https://github.com/Carkham/FedSD2C | PyTorch | NeurIPS 2024 |
| DFDG | - | - | ICDM 2024 |
| FedRevive | - | - | 2025 |
| FedBiCross | - | - | 2026 |
| FedTAD | - | - | IJCAI 2024 |
| 扩散驱动重放 | - | - | ECCV 2024 |

**注意**：大部分论文代码尚未公开或正在整理中。建议直接联系作者获取代码，或关注GitHub上的相关实现。

### 相关开源框架

1. **FedML**：https://github.com/FedML-AI/FedML
   - 支持多种联邦学习算法
   - 包含数据自由知识蒸馏实现

2. **PySyft**：https://github.com/OpenMined/PySyft
   - 隐私保护机器学习
   - 联邦学习支持

3. **Flower**：https://github.com/adap/flower
   - 联邦学习研究框架
   - 易于扩展

---

## 发展趋势与挑战

### 发展趋势

1. **生成模型融合**：
   - 扩散模型（Diffusion Models）逐渐取代GAN
   - 预训练大模型作为生成器

2. **单次通信优化**：
   - 减少通信轮次至单次
   - 提高通信效率

3. **异构性处理**：
   - 模型异构（不同架构）
   - 数据异构（非IID）
   - 系统异构（不同计算能力）

4. **持续学习结合**：
   - 联邦持续学习
   - 灾难遗忘缓解

5. **异步联邦学习**：
   - 处理过时更新
   - 提高可扩展性

### 主要挑战

1. **生成质量**：
   - 合成数据质量影响蒸馏效果
   - 模式崩溃问题

2. **通信开销**：
   - 单次通信需要传输更多信息
   - 模型压缩需求

3. **隐私-效用权衡**：
   - 更强的隐私保护可能降低模型性能
   - 差分隐私与数据自由方法的结合

4. **评估标准**：
   - 缺乏统一的评估基准
   - 不同场景下的公平比较

5. **实际部署**：
   - 边缘设备计算能力限制
   - 网络不稳定问题

---

## 参考文献

### 2025-2026年论文

1. Askin, B., Roth, H. R., Sun, Z., et al. (2025). Reviving Stale Updates: Data-Free Knowledge Distillation for Asynchronous Federated Learning. arXiv:2511.00655.

2. Xia, Y., Zhang, Y., Liu, Y., et al. (2026). FedBiCross: A Bi-Level Optimization Framework to Tackle Non-IID Challenges in Data-Free One-Shot Federated Learning on Medical Data. arXiv:2601.01901.

3. Zhang, X., Chen, Z., Yuan, Y., et al. (2025). Data-Free Continual Learning of Server Models in Model-Heterogeneous Cloud-Device Collaboration. arXiv:2509.25977.

4. Zheng, Y., Lin, B., Chen, J., & Zhu, J. (2025). HFedCKD: Toward Robust Heterogeneous Federated Learning via Data-free Knowledge Distillation and Two-way Contrast. arXiv:2503.06511.

5. Liu, J., Gao, Y., Meng, S., et al. (2025). Mosaic: Data-Free Knowledge Distillation via Mixture-of-Experts for Heterogeneous Distributed Environments. arXiv:2505.19699.

### 2024年论文

6. Zhang, J., Liu, S., & Wang, X. (2024). One-shot Federated Learning via Synthetic Distiller-Distillate Communication. NeurIPS 2024, arXiv:2412.05186.

7. Luo, K., Wang, S., Fu, Y., et al. (2024). DFDG: Data-Free Dual-Generator Adversarial Distillation for One-Shot Federated Learning. ICDM 2024, arXiv:2409.07734.

8. Liang, J., Zhong, J., Gu, H., et al. (2024). Diffusion-Driven Data Replay: A Novel Approach to Combat Forgetting in Federated Class Continual Learning. ECCV 2024 Oral, arXiv:2409.01128.

9. Wang, N., Deng, Y., Feng, W., et al. (2024). Data-Free Federated Class Incremental Learning with Diffusion-Based Generative Memory. arXiv:2405.17457.

10. Zhu, Y., Li, X., Wu, Z., et al. (2024). FedTAD: Topology-aware Data-free Knowledge Distillation for Subgraph Federated Learning. IJCAI 2024, arXiv:2404.14061.

### 早期重要工作

11. Zhu, Z., Hong, J., & Zhou, J. (2021). Data-Free Knowledge Distillation for Heterogeneous Federated Learning. ICML 2021.

12. Zhang, L., Shen, L., Ding, L., et al. (2022). Fine-tuning Global Model via Data-Free Knowledge Distillation for Non-IID Federated Learning. CVPR 2022.

13. Gao, L., Zhang, Z., & Wu, C. (2022). FedDTG: Federated Data-Free Knowledge Distillation via Three-Player Generative Adversarial Networks. arXiv:2201.03169.

---

## 附录：相关资源

### 综述论文

- A Comprehensive Survey of Federated Open-World Learning (IEEE TNNLS 2025)
- Advancing Privacy-Preserving AI: A Survey on Federated Learning and Its Applications (2025)

### 会议与期刊

- **顶级会议**：NeurIPS, ICML, ICLR, CVPR, ICCV, ECCV, AAAI, IJCAI, ICDM
- **期刊**：IEEE TNNLS, IEEE TPAMI, IEEE TMC, ACM TKDD

### 在线资源

- arXiv联邦学习最新论文：https://arxiv.org/list/cs.LG/recent
- Papers with Code - 联邦学习：https://paperswithcode.com/task/federated-learning
- Awesome Federated Learning GitHub仓库

---

*报告完成时间：2026年3月24日*
