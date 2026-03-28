# AE-SVDD Federated Robust Aggregation — Architecture Prompt

You are an expert PyTorch engineer. Your task is to implement a **two-phase AE-SVDD federated robust aggregation framework** according to the specification below. Follow every constraint precisely — deviations will cause training collapse.

---

## 1. Project Overview

A federated learning system where a central server coordinates K clients to jointly train a ResNet18 on CIFAR-10. A fraction of clients are **malicious** (attack types: Gaussian noise injection, label flipping, sign flipping, backdoor, LIE-style update attack, or user-defined). The server uses a two-phase AutoEncoder → Deep SVDD pipeline to detect and down-weight malicious model updates during aggregation.

**Key insight**: Each client uploads a full model state_dict. The server extracts BatchNorm statistics as a compact fingerprint of client behavior, then uses an AutoEncoder + SVDD to score each client's trustworthiness.

---

## 2. File Structure

```
config.py          # All hyperparameters as a dataclass with defaults
models.py          # ResNet18, Encoder (bias=False), Decoder, AutoEncoder
clients.py         # BaseClient + BenignClient + attack clients (extensible)
server.py          # FederatedServer: two-phase aggregation core logic
feature_buffer.py  # ReplayBuffer: accumulates BN features across rounds
utils.py           # BN feature extraction, robust statistics helpers
dataset.py         # CIFAR-10 IID partition, DataLoader factory
main.py            # Entry point: experiment orchestration + logging
```

---

## 3. Hyperparameters (`config.py`)

Define a `@dataclass` named `FedConfig` with the following fields and defaults:

```python
@dataclass
class FedConfig:
    # --- Federation ---
    num_clients: int = 10
    num_benign: int = 7
    total_rounds: int = 300
    attack_type: str = "gaussian_noise"   # "gaussian_noise" | "label_flipping" | "sign_flipping" | custom

    # --- Client training ---
    client_lr: float = 0.1
    client_momentum: float = 0.9
    client_weight_decay: float = 5e-4
    local_epochs: int = 1
    batch_size: int = 32

    # --- Attack params ---
    gaussian_sigma: float = 0.5
    sign_flip_scale: float = 1.0

    # --- AE / Encoder ---
    latent_dim: int = 64              # reduced from 128; see §5
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-6
    ae_grad_clip: float = 1.0

    # --- Phase schedule ---
    phase1_rounds: int = 50           # rounds 1..50 = AE warm-up
    buffer_capacity: int = 500        # replay buffer max samples

    # --- SVDD ---
    svdd_warmup_rounds: int = 100
    center_ema_decay: float = 0.9     # c_{t} = decay * c_{t-1} + (1-decay) * c_current
    tau_multiplier: float = 3.0       # threshold = median + tau * MAD
    softweight_T_start: float = 5.0
    softweight_T_end: float = 0.5
    svdd_grad_clip: float = 1.0
    svdd_recon_lambda: float = 0.1    # reconstruction regularization weight in Phase 2

    # --- Misc ---
    seed: int = 42
    device: str = "auto"              # "auto" | "cuda" | "cpu"
    data_root: str = "./data"
```

---

## 4. Model Definitions (`models.py`)

### 4.1 ResNet18

Use `torchvision.models.resnet18(weights=None, num_classes=10)`. No modifications.

### 4.2 Encoder

**CRITICAL CONSTRAINTS — violating any of these WILL cause hypersphere collapse:**

1. **ALL `nn.Linear` layers MUST have `bias=False`**. No exceptions.
2. **DO NOT use `nn.BatchNorm1d`**. Use `nn.LayerNorm` if normalization is needed.
3. **The final layer has NO activation function** (outputs raw latent vectors).
4. Keep the network **shallow** — 2 hidden layers max. The input dimension (D_bn ≈ 4480 for ResNet18) is large but we have very few samples per round.

Architecture:

```
Linear(D_bn, 256, bias=False) → LeakyReLU(0.1) → LayerNorm(256)
→ Linear(256, latent_dim, bias=False)
```

Do NOT add more layers. The AE trains on ~10 samples per round; a larger network will collapse.

### 4.3 Decoder

Mirrors the encoder. Decoder layers MAY have bias. Used only in Phase 1.

```
Linear(latent_dim, 256, bias=True) → LeakyReLU(0.1)
→ Linear(256, D_bn, bias=True)
```

### 4.4 AutoEncoder

Wraps Encoder + Decoder. Provides:
- `forward(x) → x_hat` (reconstruction)
- `encode(x) → z` (latent embedding only)

---

## 5. BN Feature Extraction (`utils.py`)

### 5.1 `extract_bn_features(state_dict) → Tensor`

From a single model state_dict:
1. Select all keys containing `"bn"` AND ending with `weight`, `bias`, `running_mean`, or `running_var`.
2. Sort keys alphabetically for deterministic ordering.
3. Flatten each tensor and concatenate → 1D vector of shape `(D_bn,)`.

### 5.2 `build_bn_matrix(client_state_dicts) → Tensor`

Stack K clients' BN features → shape `(K, D_bn)`.

### 5.3 Robust Statistics Helpers

```python
def mad(x: Tensor) → Tensor:
    """Median Absolute Deviation along dim=0, scaled by 1.4826."""
    med = x.median(dim=0).values
    return 1.4826 * (x - med).abs().median(dim=0).values

def robust_zscore(x: Tensor) → Tensor:
    """Per-feature robust z-score using median and MAD."""
    med = x.median(dim=0).values
    m = mad(x).clamp(min=1e-8)
    return (x - med) / m
```

---

## 6. Replay Buffer (`feature_buffer.py`)

### Purpose

Each round only produces K ≈ 10 BN feature vectors. Training an AE on 10 samples per step is insufficient. The buffer accumulates BN features across rounds to provide a larger training set.

### Specification

```python
class BNReplayBuffer:
    def __init__(self, capacity: int, d_bn: int)

    def add(self, X: Tensor)
        """Add K new BN feature vectors. If buffer exceeds capacity, drop oldest."""

    def sample(self, batch_size: int) → Tensor
        """Random sample from buffer. If buffer size < batch_size, return all."""

    def get_all(self) → Tensor
        """Return all buffered features."""

    def __len__(self) → int
```

### Usage in Phase 1

Each round:
1. Extract X from current client uploads.
2. Add X to buffer.
3. Sample a batch (e.g., 64~128 samples) from buffer to train AE.

This means by round 50, the buffer contains up to 500 feature vectors (50 rounds × 10 clients), giving the AE a meaningful training set.

---

## 7. Client Definitions (`clients.py`)

### 7.1 Base Interface

```python
class BaseClient(ABC):
    client_id: int
    device: torch.device

    @abstractmethod
    def local_step(self, global_state_dict: dict) → dict:
        """Receive global model, return local model state_dict."""
```

### 7.2 BenignClient

Standard SGD training on local data partition. No modifications.

### 7.3 Attack Clients

| Class | Behavior |
|---|---|
| `GaussianNoiseClient` | Add `N(0, σ²)` to global params, return directly (no training) |
| `LabelFlippingClient` | Train with flipped labels `y = 9 - y` |
| `SignFlippingClient` | Train normally, then upload `global - scale * (local - global)` |

### 7.4 Extensibility

New attack types MUST:
1. Inherit `BaseClient`.
2. Override `local_step(global_state_dict) → dict`.
3. Be registered in a factory dict in `clients.py` for config-driven instantiation:

```python
ATTACK_REGISTRY: Dict[str, Type[BaseClient]] = {
    "gaussian_noise": GaussianNoiseClient,
    "label_flipping": LabelFlippingClient,
    "sign_flipping": SignFlippingClient,
}
```

---

## 8. Server Aggregation (`server.py`) — CORE LOGIC

This is the most critical module. Follow the algorithm exactly.

### 8.0 State

```python
class FederatedServer:
    global_model: ResNet18
    ae: AutoEncoder
    buffer: BNReplayBuffer
    c: Optional[Tensor]           # SVDD center, initialized in Phase 2
    credit: Optional[Tensor]      # shape (K,), cross-round trust scores
    optimizer_ae: Adam
    config: FedConfig
```

### 8.1 Phase 1: AE Warm-up (rounds 1 .. phase1_rounds)

**Goal**: Train the AutoEncoder to reconstruct BN features, so the encoder learns a meaningful latent space BEFORE SVDD starts.

Each round:

```
1. Extract X = build_bn_matrix(client_state_dicts)     # (K, D_bn)
2. buffer.add(X)
3. X_train = buffer.sample(batch_size=min(128, len(buffer)))
4. Robust AE training step:
   a. x_hat = ae(X_train)
   b. per_sample_loss = ||x_hat - X_train||₁  along dim=1   # (N,)
   c. TRIMMED LOSS: sort per_sample_loss, discard top 20%
      keep = per_sample_loss <= quantile(per_sample_loss, 0.8)
      loss = per_sample_loss[keep].mean()
   d. optimizer_ae.zero_grad()
      loss.backward()
      clip_grad_norm_(ae.parameters(), config.ae_grad_clip)
      optimizer_ae.step()
5. Aggregate: uniform-weight FedAvg over ALL clients.
```

**Why trimmed loss**: Phase 1 includes malicious clients' BN features. Trimming the top 20% reconstruction errors prevents the AE from overfitting to malicious outliers.

**Why uniform FedAvg in Phase 1**: The AE isn't ready to filter yet. Uniform averaging is acceptable during warm-up because the global model hasn't diverged much.

### 8.2 Transition: Center Initialization (once, at round phase1_rounds + 1)

```
1. X = build_bn_matrix(client_state_dicts)
2. ae.eval()
3. Z = ae.encode(X)                                    # (K, latent_dim)
4. x_hat = ae(X)
5. recon_error = ||x_hat - X||₁ per sample             # (K,)
6. Select INIT SET: clients with recon_error below median
   init_mask = recon_error <= median(recon_error)
7. c = Z[init_mask].mean(dim=0)
8. VALIDATE: ensure no component of c is near zero
   c[c.abs() < 0.01] = 0.01
```

**Why filter by reconstruction error**: Clients that the AE reconstructs poorly are likely outliers. Using only well-reconstructed clients for center initialization avoids contamination.

### 8.3 Phase 2: SVDD Filtering (rounds phase1_rounds+1 .. total_rounds)

Each round:

```
1. Extract X = build_bn_matrix(client_state_dicts)        # (K, D_bn)
   buffer.add(X)

2. Compute embeddings (no grad):
   ae.eval()
   with torch.no_grad():
       Z = ae.encode(X)                                   # (K, latent_dim)

3. Distances to center:
   d = ||Z - c||²  per client                             # (K,)

4. Robust threshold using MAD (NOT std):
   med_d = median(d)
   mad_d = 1.4826 * median(|d - med_d|)
   mad_d = max(mad_d, 1e-6)
   threshold = med_d + config.tau_multiplier * mad_d

5. Trust mask:
   M = (d <= threshold).float()                            # (K,)
   if M.sum() < 1: M = ones(K)                            # fallback

6. Cross-round credit update:
   # Normalize current distances to [0, 1] range
   d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
   if credit is None:
       credit = d_norm
   else:
       credit = config.center_ema_decay * credit + (1 - config.center_ema_decay) * d_norm

7. Soft aggregation weights:
   p = min(1.0, svdd_round / config.svdd_warmup_rounds)
   T = T_start - p * (T_start - T_end)
   weights = exp(-credit / T) * M
   alpha = weights / (weights.sum() + 1e-12)

8. Update center c (EMA, ONLY from trusted clients):
   trusted = M > 0.5
   if trusted.sum() > 0:
       c_new = Z[trusted].mean(dim=0)
       c = config.center_ema_decay * c + (1 - config.center_ema_decay) * c_new

9. SVDD fine-tune encoder WITH reconstruction anchor (CRITICAL):
   ae.train()
   # Freeze decoder
   for p in ae.decoder.parameters():
       p.requires_grad = False

   # IMPORTANT: only encode the TRUSTED clients' features
   X_trusted = X[trusted]
   Z_trusted = ae.encode(X_trusted)           # only trusted!
   svdd_loss = ((Z_trusted - c.detach()) ** 2).sum(dim=1).mean()

   # Unfreeze decoder for reconstruction regularization
   for p in ae.decoder.parameters():
       p.requires_grad = True

   # Reconstruction anchor: sample from replay buffer (clean historical data)
   # This prevents the encoder from distorting the latent space just to
   # minimize distance to c — if it distorts too much, reconstruction breaks.
   X_buf = buffer.sample(min(64, len(buffer)))
   X_buf_hat = ae(X_buf)                      # full forward: encoder + decoder
   recon_per_sample = (X_buf_hat - X_buf).abs().mean(dim=1)
   # Trimmed: discard top 20% to avoid malicious buffer entries
   keep = recon_per_sample <= torch.quantile(recon_per_sample, 0.8)
   recon_loss = recon_per_sample[keep].mean()

   total_loss = svdd_loss + config.svdd_recon_lambda * recon_loss

   optimizer_ae.zero_grad()
   total_loss.backward()
   clip_grad_norm_(ae.parameters(), config.svdd_grad_clip)
   optimizer_ae.step()

10. Weighted aggregation:
    global_sd = weighted_fedavg(client_state_dicts, alpha)
```

### 8.4 CRITICAL IMPLEMENTATION RULES

These are **non-negotiable**. Violating any one of them will cause training collapse:

| Rule | Reason |
|---|---|
| Encoder ALL layers `bias=False` | Prevents constant-mapping collapse |
| Center c components: clamp `abs < 0.01` to `±0.01` | Prevents degenerate gradient directions |
| SVDD backward ONLY through trusted clients' features | Prevents malicious gradients poisoning encoder |
| Phase 2 SVDD loss MUST include reconstruction anchor | Prevents encoder from distorting latent space to cheat distance; anchors meaningful structure |
| Gradient clipping on ALL AE optimizer steps | Prevents gradient explosion from outlier features |
| Center update uses EMA, NEVER full replacement | Prevents center oscillation under noisy trust masks |
| Use MAD not std for thresholding | std has 0% breakdown point; MAD tolerates <50% corruption |
| Trimmed loss in Phase 1 AND in Phase 2 recon anchor | Prevents AE from learning to reconstruct malicious patterns |
| Replay buffer for AE training | 10 samples/round is insufficient; buffer provides ~500 |
| Monitor Z-space variance every round | If variance → 0, encoder is collapsing; trigger alert |

---

## 9. Evaluation & Monitoring

### 9.1 Per-Round Metrics

Log these every round:

| Metric | Source |
|---|---|
| `test_acc` | Global model on CIFAR-10 test set |
| `ae_loss` | Phase 1: trimmed reconstruction loss |
| `svdd_loss` | Phase 2: mean squared distance of trusted clients |
| `z_variance` | Variance of Z across all clients (collapse detector) |
| `center_norm` | L2 norm of c |
| `tpr` | True positive rate: fraction of malicious clients with M=0 |
| `fpr` | False positive rate: fraction of benign clients with M=0 |

### 9.2 Collapse Detection

Every round in Phase 2, check:

```python
z_var = Z.var().item()
if z_var < 1e-6:
    WARNING: encoder collapse detected
```

If collapse is detected, possible recovery: reset encoder to Phase 1 checkpoint and reduce SVDD learning rate.

### 9.3 Ground Truth

`ground_truth_labels` (1=benign, 0=malicious) are used **ONLY for monitoring** (TPR/FPR). They MUST NOT be used in any training, thresholding, or aggregation logic.

---

## 10. Baseline Comparison

Implement `aggregate_fedavg()` as a pure uniform-weight FedAvg baseline (no AE/SVDD). The main experiment script should support toggling between:
- `use_svdd=True` → full AE-SVDD pipeline
- `use_svdd=False` → FedAvg baseline

Both paths share the same clients, data partition, and evaluation, so results are directly comparable.

---

## 11. Coding Standards

- Type hints on all function signatures.
- Docstrings on all public classes and methods.
- No global mutable state; all configuration through `FedConfig`.
- `state_dict` deep-copy before distributing to clients: `{k: v.cpu().clone() for k, v in sd.items()}`.
- All tensors moved to `config.device` before computation, results moved to CPU before storage.
- Deterministic seeding: `torch.manual_seed`, `np.random.seed`, `random.seed` at startup.
