# 联邦学习恶意参与方检测论文推荐（无需根数据集）

## 核心论文推荐

### 1. FoolsGold: Secure and Private Federated Learning
- **作者**: Clement Fung, Chris J.M. Yoon, Ivan Beschastnikh
- **发表**: AISTATS 2020
- **引用**: 900+
- **核心思想**: 
  - 基于客户端更新在参数空间中的相似性来检测恶意客户端
  - **完全不需要根数据集或验证集**
  - 利用客户端梯度更新的余弦相似度，检测协同攻击的Sybil客户端
  - 为每个客户端分配信任分数，动态调整学习率
- **实验方法**:
  - **数据集**: MNIST、CIFAR-10、KDD Cup 99、Amazon Reviews
  - **攻击类型**: 标签翻转攻击、后门攻击、Sybil攻击
  - **客户端设置**: 10-100个客户端，恶意比例10%-50%
  - **非IID模拟**: 按标签划分数据
  - **评估指标**: 主任务准确率、攻击成功率、恶意客户端检测率、误报率
  - **对比基线**: FedAvg、Krum、Trimmed Mean

---

### 2. FLAME: Taming Backdoors in Federated Learning
- **作者**: Thien Duc Nguyen, Phillip Rieger, et al.
- **发表**: USENIX Security 2022
- **引用**: 640+
- **核心思想**:
  - 通过**模型聚类（HDBSCAN）**和**动态裁剪**来检测和缓解后门攻击
  - **不需要根数据集**
  - 关键观察：后门模型和良性模型在参数空间中有明显分离
  - 使用弱差分隐私（DP）添加噪声来消除残留后门
- **实验方法**:
  - **数据集**: CIFAR-10、FEMNIST、CINIC-10
  - **模型**: ResNet-18、VGG-11、MobileNet
  - **攻击类型**: 
    - 后门攻击（BadNets、Blended、Semantic）
    - 模型替换攻击
  - **客户端设置**: 50-100个客户端，恶意比例10%-30%
  - **非IID模拟**: Dirichlet分布（α=0.5）
  - **评估指标**: 
    - 主任务准确率（BA）
    - 后门攻击成功率（ASR）
    - 良性客户端保留率
  - **对比基线**: FedAvg、Krum、FoolsGold、Median、Trimmed Mean

---

### 3. DeepSight: Mitigating Backdoor Attacks in Federated Learning Through Deep Model Inspection
- **作者**: Phillip Rieger, Thien Duc Nguyen, et al.
- **发表**: NDSS 2022
- **引用**: 290+
- **核心思想**:
  - 通过**深度模型检查**来检测后门攻击
  - **不需要根数据集**
  - 两个关键组件：
    - 基于神经元激活的过滤方案
    - 使用HDBSCAN聚类识别恶意模型
  - 分析模型层权重分布的异常
- **实验方法**:
  - **数据集**: CIFAR-10、FEMNIST、IMDB、CINIC-10
  - **模型**: ResNet-18、VGG-11、LSTM
  - **攻击类型**: 
    - 多种后门攻击变体（BadNets、Blended、Clean-label）
    - 自适应攻击
  - **客户端设置**: 20-100个客户端
  - **评估指标**: 
    - 检测率（Detection Rate）
    - 误报率（False Positive Rate）
    - 主任务准确率
    - 后门成功率
  - **对比基线**: FLAME、FoolsGold、Krum、Multi-Krum

---

### 4. Learning to Detect Malicious Clients for Robust Federated Learning
- **作者**: Suyi Li, Yong Cheng, et al.
- **发表**: arXiv 2020
- **引用**: 400+
- **核心思想**:
  - 使用**元学习/检测模型**来识别恶意更新
  - 服务器训练一个检测模型来区分良性/恶意客户端更新
  - 需要少量干净数据作为训练集，但不需要完整的根数据集
- **实验方法**:
  - **数据集**: MNIST、CIFAR-10、IMDB（情感分析）
  - **模型**: CNN、LSTM
  - **攻击类型**: 
    - Byzantine攻击（标签翻转、符号翻转）
    - 目标模型投毒攻击（后门）
  - **客户端设置**: 10-100个客户端
  - **评估指标**: 
    - 测试准确率
    - 恶意客户端检测准确率
  - **对比基线**: FedAvg、Krum、Trimmed Mean

---

### 5. Krum / Multi-Krum: Machine Learning with Adversaries
- **作者**: Peva Blanchard, El Mahdi El Mhamdi, et al.
- **发表**: NIPS 2017
- **引用**: 2000+
- **核心思想**:
  - 基于**几何中位数**的聚合规则
  - **不需要根数据集**
  - Krum选择与其他梯度最接近的梯度
  - Multi-Krum选择多个梯度进行平均
  - 理论保证：在拜占庭客户端少于一半时收敛
- **实验方法**:
  - **数据集**: MNIST、CIFAR-10
  - **模型**: 逻辑回归、CNN
  - **攻击类型**: 
    - 标签翻转攻击
    - 梯度反转攻击
  - **客户端设置**: 20-100个客户端，拜占庭比例<50%
  - **评估指标**: 
    - 测试准确率
    - 收敛速度
  - **对比基线**: FedAvg、几何中位数、坐标中位数

---

### 6. Astraea: Self-balancing Federated Learning
- **作者**: Ji Gao, et al.
- **发表**: arXiv 2019
- **核心思想**:
  - 通过**统计客户端更新的分布**来检测异常
  - **不需要根数据集**
  - 使用模型更新的统计特征（L2范数、余弦相似度）
  - 动态重采样来平衡数据分布
- **实验方法**:
  - **数据集**: MNIST、Fashion-MNIST
  - **攻击类型**: 标签翻转、数据投毒
  - **客户端设置**: 10-50个客户端
  - **评估指标**: 准确率、公平性指标

---

## 实验方法总结

### 常用数据集
| 数据集 | 类型 | 特点 |
|--------|------|------|
| MNIST | 图像 | 10类手写数字，入门级实验 |
| Fashion-MNIST | 图像 | 10类服装，比MNIST更难 |
| CIFAR-10/100 | 图像 | 彩色小图像，标准基准 |
| FEMNIST | 图像 | 联邦学习专用，自然非IID |
| Shakespeare | NLP | 联邦学习基准，按角色划分 |
| IMDB | NLP | 情感分析二分类 |

### 攻击类型
1. **Byzantine攻击**
   - 标签翻转（Label Flipping）
   - 符号翻转（Sign Flipping）
   - 梯度反转（Gradient Inversion）

2. **后门攻击（Backdoor）**
   - BadNets：在图像角落添加触发器
   - Blended：将触发器与图像混合
   - Semantic：利用语义特征作为触发器
   - Clean-label：只修改特征不改变标签

3. **模型替换攻击（Model Replacement）**
   - 恶意客户端发送精心构造的模型参数
   - 旨在完全控制全局模型

### 非IID模拟方法
1. **Dirichlet分布**: 参数α控制非IID程度（α越小越非IID）
2. **按标签划分**: 每个客户端只拥有部分类别的数据
3. **数量不平衡**: 不同客户端有不同数量的样本

### 评估指标
| 指标 | 说明 |
|------|------|
| BA (Benign Accuracy) | 良性数据上的测试准确率 |
| ASR (Attack Success Rate) | 后门攻击成功率 |
| Detection Rate | 恶意客户端检测率 |
| FPR (False Positive Rate) | 良性客户端被误杀率 |
| AUC-ROC | 检测性能综合指标 |

### 常用对比基线
- **FedAvg**: 标准联邦平均
- **Krum/Multi-Krum**: 几何中位数方法
- **Median**: 坐标中位数
- **Trimmed Mean**: 截断均值
- **FoolsGold**: 基于相似性的检测
- **FLTrust**: 需要根数据集

---

## 2024-2025 最新论文推荐

### 1. FedDMC: Detecting Malicious Clients in Federated Learning (2024)
- **作者**: X Mu, K Cheng, Y Shen, et al.
- **发表**: IEEE TDSC 2024
- **引用**: 53+
- **核心思想**:
  - **无需根数据集**
  - 三个模块：客户端行为分析、异常检测、动态权重调整
  - 基于模型更新的统计特征检测恶意客户端
- **实验方法**:
  - **数据集**: MNIST、CIFAR-10、FEMNIST
  - **攻击类型**: 标签翻转、后门攻击、模型替换
  - **对比基线**: FoolsGold、FLAME、Krum、Multi-Krum
  - **评估指标**: 检测准确率、误报率、主任务准确率

---

### 2. Toward Malicious Clients Detection in Federated Learning (2025)
- **作者**: Z Dou, J Wang, W Sun, Z Liu, M Fang
- **发表**: ACM ASIACCS 2025
- **引用**: 3+
- **核心思想**:
  - **完全基于本地模型更新，无需训练数据或验证集**
  - 通过分析模型更新的统计特征和时序行为
  - 轻量级检测机制，适合资源受限环境
- **实验方法**:
  - **数据集**: CIFAR-10、CIFAR-100、Tiny-ImageNet
  - **攻击类型**: 多种后门攻击、Byzantine攻击
  - **客户端设置**: 50-100个客户端
  - **评估指标**: 检测率、计算开销、通信开销

---

### 3. Detection of Malicious Clients in Federated Learning using Graph Neural Network (2025)
- **作者**: A Sharma, N Marchang
- **发表**: IEEE Access 2025
- **引用**: 2+
- **核心思想**:
  - **无需根数据集**
  - 使用**图神经网络(GNN)**建模客户端之间的关系
  - 将客户端视为图节点，模型更新相似度作为边权重
  - 通过图结构检测异常客户端
- **实验方法**:
  - **数据集**: MNIST、Fashion-MNIST、CIFAR-10
  - **攻击类型**: 符号翻转攻击、标签翻转
  - **对比基线**: FoolsGold、Median、Trimmed Mean
  - **评估指标**: 检测准确率、F1-score、AUC-ROC

---

### 4. Efficient Backdoor Mitigation in Federated Learning with Contrastive Loss (2024)
- **作者**: H Ferguson, R Ning, J Li, H Wu, C Xin
- **发表**: OpenReview 2024
- **核心思想**:
  - **无需根数据集**
  - 使用**对比学习(Contrastive Learning)**区分良性/恶意更新
  - 利用可信的历史全局模型作为参考点
  - 通过对比损失函数净化后门属性
- **实验方法**:
  - **数据集**: CIFAR-10、GTSRB、EMNIST
  - **攻击类型**: BadNets、Blended、Clean-label后门
  - **对比基线**: FLAME、FoolsGold、DeepSight
  - **评估指标**: BA、ASR、收敛速度

---

### 5. Robust Federated Learning With Contrastive Learning and Meta-Learning (2025)
- **作者**: H Zhang, Y Chen, K Li, et al.
- **核心思想**:
  - **无需根数据集**
  - 结合**对比学习**和**元学习**
  - 解决非IID问题的同时检测恶意客户端
  - 模型级对比检测识别恶意用户模型
- **实验方法**:
  - **数据集**: CIFAR-10、CIFAR-100
  - **攻击类型**: 标签翻转、后门攻击
  - **非IID设置**: Dirichlet分布(α=0.1-0.5)
  - **评估指标**: 准确率、检测率、通信轮数

---

### 6. FedGAD: Contrastive Federated Learning for Graph Anomaly Detection (2025)
- **作者**: H Fang, Y Gao, P Zhang, S Zhou
- **发表**: IEEE 2025
- **引用**: 4+
- **核心思想**:
  - **联邦图异常检测**
  - **无需根数据集**
  - 多尺度对比学习函数
  - 协作式无监督学习
- **实验方法**:
  - **数据集**: 图数据集（如Cora、Citeseer）
  - **任务**: 图异常检测
  - **评估指标**: AUC-ROC、F1-score

---

### 7. Privacy-Preserving Federated Learning with Malicious Clients (2023)
- **作者**: J Le, D Zhang, X Lei, L Jiao, K Zeng
- **发表**: IEEE TIFS 2023
- **引用**: 103+
- **核心思想**:
  - 同时考虑**恶意客户端**和**诚实但好奇的服务器**
  - 结合差分隐私和异常检测
  - **轻量级检测机制**
- **实验方法**:
  - **数据集**: MNIST、Fashion-MNIST、CIFAR-10
  - **攻击类型**: 数据投毒、模型投毒
  - **隐私预算**: ε分析
  - **评估指标**: 准确率、隐私保护强度、检测率

---

## 2024-2025年方法趋势总结

| 趋势方向 | 代表方法 | 特点 |
|---------|---------|------|
| **对比学习** | Contrastive FL | 利用对比损失区分良性/恶意更新 |
| **图神经网络** | GNN-based | 建模客户端关系图，检测异常节点 |
| **元学习** | Meta-Learning | 快速适应不同攻击模式 |
| **多模态融合** | Multi-scale | 结合多种检测特征提高鲁棒性 |
| **轻量级检测** | Lightweight | 降低计算和通信开销 |

---

## 建议阅读顺序

### 基础阶段
1. **FoolsGold**（经典，无需根数据，易理解）
2. **Krum**（理论基础扎实）

### 进阶阶段
3. **FLAME**（聚类方法，实验详尽）
4. **DeepSight**（模型层分析）

### 最新进展
5. **FedDMC**（2024，综合检测框架）
6. **Contrastive Learning方法**（2024-2025，新思路）
7. **GNN-based方法**（2025，图视角）

## 关键论文链接

1. FoolsGold: https://arxiv.org/abs/1808.04866
2. FLAME: https://www.usenix.org/system/files/sec22-nguyen.pdf
3. DeepSight: https://arxiv.org/abs/2201.00763
4. Learning to Detect: https://arxiv.org/abs/2002.00211
5. Krum: https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
