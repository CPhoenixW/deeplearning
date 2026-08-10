# Article2 实验方案对齐与代码更新计划

## 一、目标

将 `article2` 与已确认的 AE-SVDD 实验方案对齐，在 GPU 服务器的新 Git
checkout 中验证更新后的代码，通过验收后直接提交并推送到 GitHub
`main` 分支。

## 二、已确认的实验范围

### 2.1 四数据集主实验

- 数据集：MNIST、Fashion-MNIST、CIFAR-10 和 AG News。
- 图像数据集攻击：`none`、`lf`、`gn`、`sf`、`lie`、`bd` 和 `mix`。
- `mix` 固定为 `lf+bd+gn`，恶意客户端按编号确定性分配攻击类型。
- AG News 只运行 `none`、`lf`、`gn`、`sf` 和 `lie`。
- AG News 不生成 `bd` 或包含图像触发器后门的 `mix` 任务。

### 2.2 主方法标准配置

| 参数 | 标准设置 |
| --- | ---: |
| 客户端数量 | 100 |
| 恶意客户端比例 | 30% |
| 恶意客户端数量 | 30 |
| 通信轮数 | 300 |
| 本地 epoch | 1 |
| batch size | 64 |
| 随机种子 | 42 / 43 / 44 |
| Dirichlet α_D | 1.0 |
| 服务器验证集 | 50 条可信样本 |
| Phase 1 轮数 | 15 |
| Phase 1 筛选分数 | 重建误差 |
| Phase 2 筛选分数 | SVDD distance |
| AE/SVDD 损失系数 α | 0.5 |
| descriptor 维度 | 4096 |
| center EMA | 0.9 |
| center init quantile | 0.5 |
| Phase 2 recon quantile | 0.8 |
| update clipping | 默认关闭 |

### 2.3 Phase 2 分数与训练损失分工

- Phase 1 客户端排序固定使用 reconstruction error。
- Phase 2 主方法的客户端排序仅使用 SVDD distance。
- `alpha` 只控制 Phase 2 中 `SVDD loss` 与 `reconstruction loss` 的混合比例。
- `phase2_score_mode` 与 `alpha` 必须可以独立配置，不再存在
  `recon→0`、`combined→0.5`、`svdd→1` 的隐式绑定。
- 筛选分数、训练损失和最终聚合权重必须分开记录。

### 2.4 Top-K 正式协议

- 验证候选拒绝率固定为 `0.0, 0.1, 0.2, 0.3, 0.4`。
- 必须允许 `0%` 拒绝，使无攻击或弱异常时可以保留所有客户端。
- 删除 `50%` 候选拒绝率。
- tie-breaking 仅作为固定实现规则，不作为独立论文实验。

### 2.5 攻击与对比方法

- 主实验的 8 个方法为 FedAvg、Trimmed Mean、Multi-Krum、LASA、FedSECA、
  BNGuard、FedDMC 和 AE-SVDD。
- LIE 攻击强度通过 `lie_z_override` 显式配置。
- 同一横向比较中，不同 defense 必须使用完全相同的 LIE 强度。

## 三、本次代码改造范围

1. 在 `FedConfig` 中增加独立的 Phase 2 score 配置字段，并完善参数验证。
2. 将 Phase 1 默认固定为 reconstruction score，Phase 2 默认为 SVDD distance。
3. 保留必要的旧配置兼容路径，但所有新正式配置均使用独立字段。
4. 将 Top-K 候选集改为 `0%/10%/20%/30%/40%`，并增加包含 0% 的回归测试。
5. 取消实验矩阵脚本中 score 与 `alpha` 的自动绑定。
6. 增加或改造正式实验矩阵生成器，确保：
   - 三个图像数据集生成 7 种攻击；
   - AG News 只生成 5 种允许攻击；
   - 主实验共生成 624 个 300-round 任务；
   - 随机种子、LIE 强度和标准配置被正确写入每个任务。
7. 将 SVDD 防御内部与聚合权重有关的局部变量重命名为
   `aggregation_weights`，避免与 `config.alpha` 混淆；不破坏现有 `RoundStats`
   兼容字段。
8. 每轮至少保存：
   - TACC 和可适用时的 ASR；
   - TP、FP、TN、FN、TPR、FPR、Precision 和 F1；
   - 每客户端 reconstruction score、SVDD distance 和 selection score；
   - selected rejection ratio、validation accuracy 和全部 candidate accuracies；
   - accepted client IDs 和 aggregation weights；
   - center norm、latent variance 和 center shift。
9. 增加 Stage 0 配置与相关测试，覆盖数据集、方法注册、攻击限制、验证集与
   client train set 不重叠以及有效配置元数据。

## 四、本次不执行的内容

- 不运行 624 个主实验任务。
- 不运行后续约 210 个 CIFAR-10 机制与边界实验。
- 不做 Top-K tie-breaking 敏感性实验。
- 不做 descriptor 大规模消融。
- 不做 center EMA、center quantile 和 recon quantile 的大规模扫描。
- 不删除或改写旧服务器目录中的历史实验结果。

## 五、环境与交付方式

- GitHub 仓库：`https://github.com/CPhoenixW/deeplearning.git`。
- 本计划创建时 `main` 为 `d801fbe`。
- 旧服务器目录 `/root/deeplearning/article2` 不是 Git 仓库，但包含历史日志、
  数据和虚拟环境，本次不在该目录内直接覆盖。
- 新服务器 checkout 目标：`/root/deeplearning/article2_github_20260810`。
- 本机隔离 checkout 用于编辑、Git diff 审查和提交。
- 服务器新 checkout 用于复用已有数据、Python 环境和 GPU 进行最终验证。
- 验收通过后直接提交并推送到 GitHub `main`。
- 禁止 force push。如果 `main` 存在分支保护或出现新的远程提交，停止推送并报告，
  不覆盖他人更新。

## 六、验收与测试矩阵

### 6.1 静态检查

```bash
python -m compileall article2/src article2/tools article2/tests
```

### 6.2 单元和回归测试

- 现有 `article2/tests` 全量测试。
- Top-K 候选集精确等于 `0.0/0.1/0.2/0.3/0.4`。
- 0% 拒绝率在验证最优时确实可被选中。
- Phase 1 始终使用 reconstruction score。
- Phase 2 的 `recon/svdd/combined` score 可在 `alpha=0.5` 下独立切换。
- `alpha=0.25/0.5/0.75` 可在 Phase 2 score 固定为 SVDD 时独立切换。
- 主实验生成器精确生成 624 个任务，且 AG News 不含 `bd/mix`。
- 服务器验证集与客户端训练样本索引不重叠。

### 6.3 服务器 Stage 0 短测

- 每个任务 10–20 轮，只使用 1 个随机种子。
- 使用代表性任务覆盖 4 个数据集、8 个主实验方法和允许的攻击路径，不运行
  26×8 的全笛卡尔积。
- 核对 AG News 不生成 `bd/mix` 任务。
- 核对 Top-K 实际记录包含 0% 候选。
- 核对 Phase 1/Phase 2 score、`alpha=0.5`、验证集大小和 LIE 强度的有效配置。
- 所有 Stage 0 任务都必须正常结束并生成可解析的结果元数据。

## 七、性能检查

本次不对模型收敛性做最终论文结论，但 Stage 0 将记录：

- 每个短任务的端到端时间；
- 服务器防御阶段是否出现明显性能回归；
- GPU 执行是否出现显存溢出、NaN/Inf 或不可恢复的聚合失败。

由于旧目录和 GitHub 新版本的代码基线不同，不使用历史 100-round 实验作为严格性能
baseline；本次主要检查是否出现明显的新退化。

## 八、风险和回滚

- 旧 `/root/deeplearning/article2` 完整保留，可通过重新选择该目录立即回滚。
- 不删除历史日志、数据集或虚拟环境。
- 新 checkout 可以通过 Git commit 回滚，但本次不使用破坏性 Git 命令。
- Stage 0 中的功能性错误、配置错误、NaN/Inf、日志字段缺失或任务生成数不符都会阻止
  推送。
- 如果服务器直连 GitHub 仍出现 TLS 错误，通过本机已认证的 GitHub 会话完成提交与推送，
  不在服务器上保存 GitHub token。
