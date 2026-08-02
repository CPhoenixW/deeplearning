# 阶段 A：任务模型训练参数校准

## 目标与选择规则

阶段 A 只运行无攻击 FedAvg，即 `attack=none`、`defense=avg`。分别为 MNIST、Fashion-MNIST、CIFAR-10 和 AG News 选择：

- `client_lr ∈ {0.005, 0.01, 0.03, 0.05, 0.1}`
- `client_weight_decay ∈ {0, 1e-4, 5e-4, 1e-3}`

排序指标只能是干净测试准确率 TACC。单次运行取最后 10 轮 TACC 均值；多 seed 确认阶段再对各 seed 的该均值求平均。运行时间、防御检测率和攻击指标不得参与选择。选定后，应把每个任务的 `client_lr` 与 `client_weight_decay` 写入 `configs/hyperparameters.json` 的 `tasks` 段，并对 FedAvg、所有对比防御和 AE-SVDD 保持一致。

## 与三张 RTX 6000D 匹配的执行方案

服务器有三张约 85.7GB 的 RTX 6000D、128 个 CPU 线程和 1TB 内存。候选组合彼此独立，因此采用“一张 GPU 一个 trial”的三 worker 动态队列；不使用跨卡 DDP。单个 trial 内继续使用现有 CUDA vmap 批量客户端执行器，提高小模型的 GPU 利用率。

固定协议：K=100、无恶意客户端、IID、local epoch=1。筛选和确认阶段都固定对应任务的 batch size 与客户端批量组大小：

| 任务 | batch size | client batch group size |
|---|---:|---:|
| MNIST | 256 | 25 |
| Fashion-MNIST | 256 | 25 |
| CIFAR-10 | 128 | 10 |
| AG News | 128 | 10 |

不启用 AMP，因为当前批量客户端执行器与 AMP 互斥。每个 GPU worker 固定 8 个 CPU 数学线程，避免三个进程同时占满 128 线程。

### A1：全候选筛选

运行 4×5×4×1=80 个 trial，seed=42，每个 trial 60 轮。按最后 10 轮干净 TACC 均值为每个任务保留前三名。60 轮结果只用于减少候选，不能作为最终超参数结论。

```bash
python3 -m tools.stage_a_calibration \
  --manifest configs/stage_a_screen.json plan

nohup python3 -m tools.stage_a_calibration \
  --manifest configs/stage_a_screen.json run \
  > log/stage_a/screen_launcher.log 2>&1 &
```

调度器支持断点续跑：重新执行相同命令时，会检查最终 JSON 的轮数、seed、LR 和 weight decay，跳过已完成 trial。

全部完成后生成排名：

```bash
python3 -m tools.stage_a_calibration \
  --manifest configs/stage_a_screen.json select
```

### A2：全预算确认

将每个任务的前三名提升到 300 轮，并用 seeds 42、43、44 确认，共 4×3×3=36 个 trial：

```bash
python3 -m tools.stage_a_calibration \
  --manifest configs/stage_a_screen.json promote \
  --output configs/stage_a_confirm.json

python3 -m tools.stage_a_calibration \
  --manifest configs/stage_a_confirm.json plan

nohup python3 -m tools.stage_a_calibration \
  --manifest configs/stage_a_confirm.json run \
  > log/stage_a/confirm_launcher.log 2>&1 &

python3 -m tools.stage_a_calibration \
  --manifest configs/stage_a_confirm.json select
```

最终结果位于 `log/stage_a/confirm_selection.json`。其中 `recommended_hyperparameters_patch.tasks` 可复制进统一超参数文件；在后续攻击和防御实验中不得再针对某个防御单独调整这两个训练参数。

## 服务器首次准备

服务器已有 PyTorch 2.7、CUDA 12.8 和 torchvision 0.22。AG News 还需要 HuggingFace datasets：

```bash
cd /root/deeplearning/article2
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install datasets==4.8.4

# 串行准备数据，避免三个 worker 同时下载同一数据集
python -m tools.stage_a_calibration \
  --manifest configs/stage_a_screen.json prepare-data
```

`prepare-data` 会通过 `prepare_data_overrides` 临时允许下载 AG News；生成的正式 trial 保持 `hf_datasets_offline=true`，只读取已经验证的缓存，不会因外网波动改变实验。

查看进度：

```bash
tail -f log/stage_a/screen_launcher.log
nvidia-smi
```

每个 trial 的完整控制台输出保存在各自结果目录的 `console.log`；最终 JSON 的 `meta.effective_config` 保存实际生效的全部参数，便于审计。
