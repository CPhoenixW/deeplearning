# FedDMC 攻击复现对照

依据 Mu 等人在 *FedDMC: Efficient and Robust Federated Learning via Detecting
Malicious Clients*（TDSC 2024）第 V-B 节，本文框架可将其攻击设置对应如下。

| 论文攻击 | 当前 ID | 复现结论 | 实现位置 / 说明 |
| --- | --- | --- | --- |
| Label Flipping (LF) | `lf` | 可复现 | 恶意客户端把每个标签均匀随机替换为任一其他类别；本次改为与论文描述一致的随机非原类翻转。 |
| LIT attack | `lie` | 可复现 | 以已知良性更新为输入，在轮末依论文的 `s` 和高斯分位数 `z_max` 统一重写恶意上传。 |
| Gaussian (GS) | `gn` | 可复现 | 对每个浮点层按当前全局层参数的均值和方差采样高斯替代上传。 |
| Scaling attack | `bd` | 条件复现 | 使用触发器、目标标签和模型替换缩放；可测 ASR。论文没有给出其“检测器允许范围”的具体约束/优化器，因此这里实现的是可执行的标准缩放后门，而非声称逐式等价。 |
| Adaptive attack（额外鲁棒性试验） | — | 暂不作为精确复现 | 论文只给出带欧氏距离与维度差异项的目标，未给出有界可行域、`d` 的选择规则、优化步骤或超参数。无这些信息目标不能唯一确定，需作者代码或补充材料后再实现。 |

说明：`minmax` 与 `minsum` 是框架中已有的 Shejwalkar & Houmansadr (2021)
攻击；它们不是 FedDMC 的四个默认攻击，但可用于补充评估距离约束下的模型投毒。

建议使用与论文相同的约 28% 恶意客户端比例，并在每个攻击下分别记录检测率、
测试准确率；对 `bd` 还应记录 `backdoor_asr`。例如：

```bash
python -m src.pipeline --config configs/primary_matrix.json --dry-run
```
