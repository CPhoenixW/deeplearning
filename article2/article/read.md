## Fed DMC
二叉决策树分类恶意参与方

无根数据集，不用进行全局的模型检验

实验过程

1.多数据集

MNIST, EMINST, Cifar10, COVIDx

2.多种攻击

Lable-flipping, Scaling attack

LIT, Gauss, Adaptive


3.选择相同不需干净验证集的方法进行比较

Multy-Krum, Auror, FoolsGold, FLDetector

4.模块消融实验

5.参数敏感性分析

6.计算效率对比

## FLAD, SMTFL 
需要干净验证集




## Experiments

## Experimental Setup
### Datasets and global-model architectures


**MNIST**[1] is a 10-class digital image classification dataset consisting of 60,000 training examples and 10,000 test examples. For MNIST, we use a fully connected network (FC) with layer sizes {784, 100, 10} as the global model architecture.

**FashionMNIST**  

**Cifar10** is a 10-class color image classification dataset consisting of a predefined set of 50,000 training and 10,000 test examples. Each class has an equal number of samples, i.e., 6,000 images per class. For CIFAR10, we utilized the widely used ResNet18 architecture as the
global model.

**COVIDx** is a large-scale, multinational chest radiograph dataset, containing over 30,000 images from 16,400 patients. We also utilized ResNet50 architecture as the global model.

**HAR**


攻击手段, Lable-flipping, Scaling attack, LIE, Gauss, Backdoor, 混合
防御手段, Fedavg, svdd, Multy-Krum, Auror, FoolsGold, FLDetector, FLTrust
评估指标, DACC, FPR/FNR, AAR; TACC, ASR
默认配置, 

B. 实验结果
参数敏感性分析： 硬间隔参数分析，模型退火轮数参数分析。
恶意参与方比例
主实验： 多数据集，多攻击，多防御对比
消融实验： 
适应性攻击： 
效率分析： 


C. Impact of Key Parameters



[1] Y. LeCun. 1998. The MNIST database of handwritten digits. http://yann. lecun.
com/exdb/mnist/.







您设计的实验框架**整体逻辑非常严密，符合目前联邦学习安全/鲁棒聚合类顶会论文的标准结构**。尤其是您将攻击分为模型投毒和数据投毒，并加入了自适应攻击和参数敏感性分析，这都是审稿人非常看重的点。

不过，在深入细节时，我有**几个关键的改进建议（排雷）**，并为您试写了 `1. Experimental Setup` 的核心内容，您可以直接参考或修改后用于论文。

---

### 一、 框架的改进建议（重要排雷）

1.  **Dirichlet参数 $\beta = 5$ 太高了，不属于Non-IID**：
    *   **雷点**：在基于 Dirichlet 分布 $Dir(\beta)$ 的数据划分中，$\beta \to \infty$ 代表完全独立同分布（IID）。$\beta = 5$ 时，各个客户端拥有的各类别数据比例已经非常均匀了，**无法体现 Non-IID 场景的挑战性**。
    *   **建议**：默认设置（Default Setting）应改为 **$\beta = 0.5$**（代表中度 Non-IID）或 **$\beta = 0.1$**（代表高度 Non-IID）。然后在 `2.2 Impact of the Different Degrees of Non-IID` 中，再测试 $\beta \in \{0.1, 0.5, 1.0, 5.0\}$。
2.  **网络模型过载（Overkill）**：
    *   **雷点**：用 ResNet18 跑 MNIST 会极其容易过拟合，且计算资源的浪费在审稿人看来显得不够专业。
    *   **建议**：MNIST 使用简单的 2层 CNN 或多层感知机 (MLP)；FashionMNIST 使用 3层 CNN；CIFAR-10 使用 ResNet-18。
3.  **数据集的选择 (COVIDx 和 HAR)**：
    *   HAR (Human Activity Recognition) 是很好的选择，能展示算法在传感器/时序一维数据上的通用性（对应 1D-CNN 或 MLP）。
    *   COVIDx（胸透X光）作为医学图像数据非常有现实意义，但如果你想体现多模态，建议将 COVIDx 更换为文本数据集（如 **AG News** 或 **IMDB**，使用 TextCNN 或轻量级 Transformer），这样你的论文就涵盖了**基础图像、复杂图像、传感器数据和文本数据**，说服力拉满。
4.  **评价指标缩写的规范化**：
    *   您的 DAR, DPR, RR 应该是 Detection Accuracy Rate, Detection Precision Rate, Recall Rate（或 Rejection Rate）。建议在公式中明确对齐标准的异常检测指标：**TPR (True Positive Rate/Recall)** 和 **FPR (False Positive Rate)**，因为 FPR 是衡量“误杀正常客户端”的核心指标。

---

### 二、 Experimental Setup 内容填充（可作为论文 Draft）

下面为您用标准学术英语撰写/搭建了第一部分的内容，您可以根据具体的实验细节填空。

#### 1.1 Datasets and Global-Model Architectures
To comprehensively evaluate the effectiveness of our proposed SVDD-based hard isolation method, we conduct experiments across four diverse datasets, encompassing different modalities and complexities:
*   **MNIST & FashionMNIST**: Used for fundamental image classification tasks. We employ a lightweight Convolutional Neural Network (CNN) consisting of two convolutional layers and two fully connected layers.
*   **CIFAR-10**: A complex colored image dataset. We adopt the standard **ResNet-18** architecture to evaluate the defense performance on deep models.
*   **UCI-HAR**: A Human Activity Recognition dataset derived from smartphone sensors, representing time-series/1D data. A 1D-CNN model is applied.
*(如果换成文本数据：)*
*   **AG News**: A text classification dataset used to demonstrate the generalizability of our algorithm in Natural Language Processing (NLP) tasks. A TextCNN architecture is deployed.

**Data Heterogeneity (Non-IID Setting):**
To simulate realistic federated learning environments without a root dataset, we distribute the training data among clients following a Dirichlet distribution $Dir(\beta)$. In our default setting, we set the concentration parameter **$\beta = 0.5$** to generate a highly skewed Non-IID data distribution, which makes distinguishing between malicious updates and benign Non-IID deviations highly challenging. The impact of different $\beta$ values ($\beta \in \{0.1, 0.5, 1.0, 5.0\}$) is further analyzed in Section 2.2.

#### 1.2 Attack Methods
We assume a fraction of clients are malicious. The attacks are categorized into two primary paradigms to thoroughly test our single-class hard isolation mechanism:

**1) Data Poisoning Attacks:**
*   **Label-Flipping Attack (LF):** Malicious clients systematically flip the labels of their local training data (e.g., $l_{poisoned} = 9 - l_{true}$ for CIFAR-10) to degrade the global model's accuracy.
*   **Backdoor Attack:** Attackers inject a specific trigger (e.g., a localized pixel patch) into a subset of training images and change their labels to a target class. The goal is to manipulate the model's prediction on test samples containing the trigger while maintaining normal accuracy on clean data.

**2) Model Poisoning Attacks:**
*   **Gaussian Noise Attack (Gauss):** Attackers replace their local model updates with random noise drawn from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$, aiming to disrupt the global convergence.
*   **Scaling Attack:** Malicious clients scale their local updates by a large factor $\gamma$ (e.g., multiplying gradients by 50) to dominate the global model aggregation.
*   **A Little Is Enough (LIE):** A sophisticated omniscient attack where the adversary estimates the mean and variance of benign updates, and adds carefully crafted noise to shift the global model without exceeding the statistical threshold of traditional defenses.
*   **Random Combination:** A hybrid scenario where malicious clients randomly select one of the aforementioned attack strategies in each communication round, mimicking an uncoordinated, realistic threat environment.

#### 1.3 Robust Aggregation Methods (Baselines)
We compare our proposed method with **FedAvg** (the standard baseline without defense) and five state-of-the-art Byzantine-robust aggregation methods. Notably, to ensure a fair comparison, **all selected baselines do not rely on an auxiliary/root dataset on the server**:
*   **Multi-Krum:** A classic distance-based defense that selects a subset of local models with the minimal sum of squared Euclidean distances to their nearest neighbors.
*   **FLDetector (CVPR 2022):** Detects malicious clients by modeling the temporal consistency of local model updates over multiple rounds, independently of server-side data.
*   **DnC (Divide-and-Conquer, NDSS 2021):** Employs Principal Component Analysis (PCA) on the covariance matrix of client updates to filter out malicious gradients projecting onto principal components.
*   *(推荐加入的新开源算法1)* **LASA (WACV 2025):** A recently proposed layer-adaptive sparsified model aggregation algorithm that leverages gradient sparsity features to filter malicious updates in Non-IID settings.
*   *(推荐加入的新开源算法2)* **BRAFL (IJCNN 2025):** Utilizes median absolute deviation and DBSCAN clustering with cosine distance to securely aggregate models under extreme data heterogeneity.
*   **Ours (SVDD-based Hard Isolation):** Our proposed method mapping client updates into a high-dimensional feature hypersphere via Support Vector Data Description, dynamically establishing a hard decision boundary to isolate anomalous updates without assuming prior knowledge of the attack distribution.

#### 1.4 Evaluation Metrics
We comprehensively evaluate the performance using metrics across two dimensions: model utility and defense capability.
*   **Test Accuracy (TACC):** The classification accuracy of the final aggregated global model on the clean testing dataset.
*   **Attack Success Rate (ASR):** Specifically used for Backdoor attacks, measuring the proportion of target-class predictions on backdoor-triggered testing samples.
*   **True Positive Rate (TPR / Recall Rate):** The ratio of successfully detected malicious clients out of all actual malicious clients. (对应您的 RR/DAR)
*   **False Positive Rate (FPR):** The ratio of benign clients mistakenly identified as malicious. A lower FPR indicates better protection of benign Non-IID updates. (用以替代单纯的 DPR，在检测领域TPR/FPR是最权威的描述)
*   **F1-Score:** The harmonic mean of Precision and Recall for malicious client detection, providing a single metric for overall detection performance.

---

### 三、 给您算法（SVDD单分类硬隔断）的额外写作建议

既然您的算法是基于 **单分类算法 (One-class classification, 如 SVDD)** 的硬隔断：
1.  **最大卖点**：传统的聚类算法（如K-means, DBSCAN）在应对投毒时，如果恶意节点比例超过良性节点，聚类就会被恶意节点“带偏”（因为最大的簇变成了恶意簇）。而 SVDD 是寻找包含大多数正常样本的最小超球体，只要我们在时间维度（历史信任度）上稍微加以约束，它对**高比例恶意节点（>50%）**的抵抗力可能会远超传统算法。
2.  **在 2.2.1 Impact of the Number of Malicious Clients 必须重点展示**：把恶意比例拉到 40% 甚至 60%，展示基线算法崩溃，但您的 SVDD 因为硬隔断超球体的存在，依旧能圈住良性模型。
3.  **在 2.2.3 压缩维度的影响 (Impact of different compression dimensions)**：这个设计非常聪明。因为模型参数动辄几百万维，直接做 SVDD 会遭遇“维度灾难 (Curse of Dimensionality)”。您肯定是先做了降维（比如AutoEncoder或PCA）再做SVDD。这个消融实验能很好地解释您的算法如何克服计算开销问题。

祝您的论文实验顺利！如果需要针对某一种特定攻击（如LIE）的代码实现建议，也可以随时告诉我。
2 Experimental Result
2.1 Global Performance
1）Performance of the global models
2）Detection results

2.2 Impact of Key Parameters
1）Impact of the Number of Malicious Clients
2）Impact of the Different Degrees of Non-IID
3）impact of different compression dimensions
4）Impact of hyperparameters
5）Adaptive attack and impact of the detection iteration