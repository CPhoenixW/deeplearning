# 3. Proposed Method

This section presents a server-side defense method for poisoning attacks in federated learning, termed **AE-SVDD**. The method uses only the client model states received in each communication round and does not access raw server-side training data or trusted validation samples. Its main idea is to first compress the complete, high-dimensional, and heterogeneous set of trainable parameters into a fixed-dimensional layer-aware representation. It then learns the distribution of normal clients in the ongoing communication process through two-stage autoencoder-SVDD anomaly modeling. Finally, it performs hard filtering using MAD-based thresholds without requiring prior knowledge of the number of malicious clients, and aggregates only the retained client models.

## 3.1 Framework Overview

Consider a synchronous federated learning system consisting of a server and $K$ clients. In communication round $t$, client $i$ performs local training starting from the model distributed by the server and uploads its model state $\Theta_i^t$. Let $\boldsymbol{\theta}_i^t=\{\boldsymbol{\theta}_{i,\ell}^t\}_{\ell=1}^{L}$ denote all $L$ trainable parameter tensors contained in $\Theta_i^t$. This set of parameters is the sole input to the subsequent representation and detection modules.

An adversary may launch data-poisoning or model-poisoning attacks by tampering with local data, local training procedures, or uploaded models. The server does not know the attack type, the proportion of malicious clients, or their identities, and does not retain trusted data for client scoring. Rather than assuming theoretical security guarantees under malicious-majority or adaptive-attack settings, the proposed method evaluates detection and aggregation robustness experimentally under these unknown attack conditions.

For each uploaded model, the server first constructs a $D=4096$-dimensional layer-aware descriptor using a fixed mapping $\Phi(\cdot)$. It then applies feature-wise median/MAD normalization across the clients in the current round to obtain $\mathbf{x}_i^t\in\mathbb{R}^{D}$. During Phase I, which spans the first $T_{\mathrm{w}}$ rounds, the server uses reconstruction error alone to characterize anomalies. Before a reliable normal region has been established in the latent space, this process yields a relatively conservative set of accepted clients for training the autoencoder. In the final round of Phase I, the latent representations of accepted clients are used to initialize the SVDD center. The method then enters Phase II, where reconstruction error and squared distance to the center are converted into comparable robust standardized scores and fused to jointly capture two types of anomalies: samples that are difficult to reconstruct and samples that deviate from the normal latent region.

In both phases, anomaly scoring is completed before the autoencoder and center are updated in the current round. Only retained clients with finite values are allowed to participate in subsequent parameter learning and global model aggregation. This detect-then-update ordering prevents rejected clients from directly influencing the normal representation in the same round. Section 3.4 describes the MAD-based decision rule and hard-filtered aggregation scheme shared by both phases.

## 3.2 Full-Parameter Layer-Aware Representation

Directly concatenating all client parameters produces vectors with millions or even more dimensions. This not only increases server-side computational overhead, but may also cause localized anomalous changes to be overwhelmed by irrelevant coordinates. Therefore, we map all trainable parameters of each client into a fixed 4096-dimensional layer-aware descriptor. This mapping is determined by a fixed random seed, hash buckets, and signs. Consequently, all clients and all communication rounds share exactly the same feature coordinate system, and the server does not need to explicitly store a dense $4096\times|\boldsymbol{\theta}|$ projection matrix.

Since the implementation uses a zero tensor as the reference, the parameter difference is

$$
\Delta\boldsymbol{\theta}_{i,\ell}^{t}
=\boldsymbol{\theta}_{i,\ell}^{t}-\mathbf{0}
=\boldsymbol{\theta}_{i,\ell}^{t},
$$

which is exactly the client's absolute trainable parameter tensor.

### 3.2.1 Fixed Layer-Aware CountSketch

Let $P_{\ell}$ denote the number of elements in the $\ell$-th trainable parameter tensor. The server partitions the 4096 output coordinates into pairwise disjoint layer-wise subspaces $\{\mathcal{B}_{\ell}\}_{\ell=1}^{L}$. Each parameter tensor receives at least one coordinate, while the remaining coordinates are allocated in proportion to $\sqrt{P_{\ell}}$. Thus, large parameter tensors receive greater representation capacity without linearly occupying almost the entire feature budget merely because they contain more parameters.

For each parameter coordinate $p$ in tensor $\ell$, the tensor name and fixed seed determine a bucket mapping $h_{\ell}(p)\in\mathcal{B}_{\ell}$ and a sign mapping $s_{\ell}(p)\in\{-1,+1\}$. The $j$-th coordinate of the final descriptor $\boldsymbol{\phi}_i^t\in\mathbb{R}^{4096}$ is

$$
\phi_{i,j}^{t}
=\sum_{\ell=1}^{L}
\sum_{p:h_{\ell}(p)=j,\;j\in\mathcal{B}_{\ell}}
s_{\ell}(p)\Delta\theta_{i,\ell,p}^{t},
\qquad j=1,\ldots,4096.
$$

In other words, multiple parameters within the same tensor may be randomly assigned to the same bucket and accumulated with signs. However, under the default 4096-dimensional configuration, different parameter tensors occupy different descriptor segments. This layer-aware constraint enables the anomaly model to compare the overall parameter behavior of each client while preserving clues about the parameter groups in which anomalies primarily occur. If the output dimension is smaller than the number of parameter tensors, the implementation falls back to a shared hash space; the main 4096-dimensional setting in this work does not rely on this fallback.

### 3.2.2 Robust Normalization

The numerical scales of different parameter coordinates may differ by several orders of magnitude, while means and standard deviations are susceptible to malicious uploads. Therefore, in each round and for each descriptor coordinate $j$, the server normalizes values using the median and MAD over the $K$ clients:

$$
\mu_j^t=\operatorname{median}_{i}(\phi_{i,j}^{t}),\qquad
r_j^t=\operatorname{median}_{i}\left|\phi_{i,j}^{t}-\mu_j^t\right|,
$$

$$
x_{i,j}^{t}
=\frac{\phi_{i,j}^{t}-\mu_j^t}
{\max(1.4826r_j^t,\epsilon)}.
$$

Here, $1.4826$ makes the MAD comparable in scale to the standard deviation under a Gaussian assumption, and $\epsilon>0$ prevents division by zero for constant coordinates. The implementation maintains a finite-value mask for each descriptor row. Non-finite descriptor rows are replaced with safe values only during normalization to maintain numerical stability; their subsequent anomaly scores are set to $+\infty$, so they cannot be accepted or aggregated.

## 3.3 Two-Stage AE-SVDD Anomaly Modeling

Let $f_{\psi}(\cdot)$ and $g_{\omega}(\cdot)$ denote the encoder and decoder, respectively. For the normalized representation $\mathbf{x}_i^t$, its latent vector and reconstruction are

$$
\mathbf{z}_i^t=f_{\psi}(\mathbf{x}_i^t),\qquad
\widehat{\mathbf{x}}_i^t=g_{\omega}(\mathbf{z}_i^t).
$$

The current implementation uses a shallow autoencoder. Its encoder has the architecture $4096\rightarrow256\rightarrow d_z$, with LeakyReLU and LayerNorm between layers. Its decoder has the architecture $d_z\rightarrow256\rightarrow4096$, with LeakyReLU as the intermediate activation. The reconstruction error for a single client is defined as the mean absolute error:

$$
e_i^t=\frac{1}{D}\left\|\widehat{\mathbf{x}}_i^t-\mathbf{x}_i^t\right\|_1.
$$

### 3.3.1 Phase I: Reconstruction Warm-up

During the warm-up stage, $t\leq T_{\mathrm{w}}$, the autoencoder has not yet formed a stable normal region in latent space. Therefore, only $e_i^t$ is used as the anomaly score:

$$
a_i^t=e_i^t.
$$

This score is passed to the MAD rule in Section 3.4 to obtain the accepted set $\mathcal{A}_t$. The server then updates the autoencoder using only clients in $\mathcal{A}_t$:

$$
\mathcal{L}_{\mathrm{P1}}^t
=\frac{1}{|\mathcal{A}_t|}
\sum_{i\in\mathcal{A}_t} e_i^t.
$$

Thus, the warm-up stage does not assume that all early clients are benign. Instead, the filter-then-learn procedure progressively reduces contamination of the reconstruction model by anomalous samples.

### 3.3.2 SVDD Center Initialization

After the autoencoder update in the final round of Phase I, the center is initialized using the updated latent representations of accepted clients from the same round:

$$
\mathbf{c}^{T_{\mathrm{w}}}
=\frac{1}{|\mathcal{A}_{T_{\mathrm{w}}}|}
\sum_{i\in\mathcal{A}_{T_{\mathrm{w}}}}
f_{\psi}(\mathbf{x}_i^{T_{\mathrm{w}}}).
$$

To prevent degeneration of individual center coordinates, the implementation sets coordinates satisfying $|c_j|<0.01$ to $0.01$. This initialization ensures that the center used in the first round of Phase II lies in the same latent space as the warmed-up autoencoder.

### 3.3.3 Phase II: Joint Reconstruction and SVDD Modeling

When $t>T_{\mathrm{w}}$, the server first scores clients using the autoencoder parameters at the beginning of the current round and the latent center from the previous round, $\mathbf{c}^{t-1}$. For client $i$, the SVDD distance is defined as

$$
d_i^t=\left\|\mathbf{z}_i^t-\mathbf{c}^{t-1}\right\|_2^2.
$$

The numerical ranges of reconstruction error $e_i^t$ and latent distance $d_i^t$ may vary substantially with training progress, data heterogeneity, and model state. To make these two sources of anomaly evidence comparable on a robust scale, the server calculates their locations and scales separately over the set of clients with finite scores in the current round, $\mathcal{F}_t$:

$$
m_e^t=\operatorname{median}_{i\in\mathcal{F}_t}(e_i^t),\qquad
r_e^t=\operatorname{median}_{i\in\mathcal{F}_t}\left|e_i^t-m_e^t\right|,
$$

$$
m_d^t=\operatorname{median}_{i\in\mathcal{F}_t}(d_i^t),\qquad
r_d^t=\operatorname{median}_{i\in\mathcal{F}_t}\left|d_i^t-m_d^t\right|.
$$

The two scores are then standardized using their respective median and MAD:

$$
\widetilde{e_i^t}=
\frac{e_i^t-m_e^t}
{\max(1.4826r_e^t,\epsilon)},
\qquad
\widetilde{d_i^t}=
\frac{d_i^t-m_d^t}
{\max(1.4826r_d^t,\epsilon)}.
$$

The constant $1.4826$ makes the MAD comparable to the standard deviation under an approximately Gaussian distribution, while $\epsilon>0$ avoids numerical instability when the MAD is zero. The final joint anomaly score is defined as

$$
a_i^t=\widetilde{e_i^t}+\widetilde{d_i^t}.
$$

This score is not restricted to $[0,1]$. Its value indicates the robust degree to which a client deviates from typical behavior in the current round. A larger $\widetilde{e_i^t}$ indicates that the client descriptor is difficult for the current autoencoder to reconstruct, while a larger $\widetilde{d_i^t}$ indicates that its latent representation deviates from the normal center. Their sum ensures that a client obtains a low joint anomaly score only when it exhibits low deviation according to both sources of evidence. If a client's descriptor, reconstruction error, or latent distance is non-finite, its joint score is set to $+\infty$, preventing it from passing the subsequent filter.

After obtaining the accepted set $\mathcal{A}_t$ using the MAD thresholding rule in Section 3.4, the server updates the latent center using only retained clients:

$$
\overline{\mathbf{c}}^t=
\frac{1}{|\mathcal{A}_t|}
\sum_{i\in\mathcal{A}_t}\mathbf{z}_i^t,
\qquad
\mathbf{c}^{t}
=\rho\mathbf{c}^{t-1}+
(1-\rho)\overline{\mathbf{c}}^t,
$$

where $\rho\in[0,1)$ is the exponential moving average coefficient for the center. After the update, center coordinates with absolute values below $0.01$ are set to $0.01$ to avoid degeneration in individual latent dimensions.

The Phase II objective simultaneously contracts accepted clients toward the center in latent space and preserves their descriptor reconstruction capability:

$$
\mathcal{L}_{\mathrm{svdd}}^t
=
\frac{1}{|\mathcal{A}_t|}
\sum_{i\in\mathcal{A}_t}
\left\|f_{\psi}(\mathbf{x}_i^t)-\mathbf{c}^{t}\right\|_2^2,
$$

$$
\mathcal{L}_{\mathrm{rec}}^t
=
\frac{1}{|\mathcal{Q}_t|}
\sum_{i\in\mathcal{Q}_t} e_i^t,
\qquad
\mathcal{Q}_t=
\left\{
i\in\mathcal{A}_t:
e_i^t\leq q_{\eta}
\right\},
$$

where $q_{\eta}$ denotes a predefined lower-quantile threshold for reconstruction errors within the accepted set. This additional truncation causes the reconstruction branch to use only accepted clients with relatively low reconstruction errors, reducing the influence of boundary clients on autoencoder updates. The final objective is

$$
\mathcal{L}_{\mathrm{P2}}^t
=
\lambda\mathcal{L}_{\mathrm{svdd}}^t+
(1-\lambda)\mathcal{L}_{\mathrm{rec}}^t,
\qquad \lambda\in[0,1].
$$

When computing $\mathcal{L}_{\mathrm{svdd}}^t$, the decoder parameters are temporarily frozen so that the SVDD branch primarily contracts the latent representations produced by the encoder. The reconstruction branch continues to jointly update the encoder and decoder. Scoring, filtering, center updating, and autoencoder updating are strictly performed in this order. Consequently, rejected clients neither participate in global model aggregation nor influence the autoencoder parameters or SVDD center.

## 3.4 Adaptive MAD-Based Client Filtering

The proposed method does not predefine the number of malicious clients. Instead, it treats the anomaly scores $\{a_i^t\}_{i=1}^{K}$ in the current phase as a one-dimensional robust anomaly detection problem. For all finite scores, the server computes

$$
m_t=\operatorname{median}_{i}(a_i^t),\qquad
\operatorname{MAD}_t
=\operatorname{median}_{i}|a_i^t-m_t|.
$$

Given a nonnegative coefficient $\kappa$, the adaptive threshold is

$$
\tau_t=m_t+\kappa\operatorname{MAD}_t.
$$

The client decision is

$$
b_i^t=
\begin{cases}
1, & a_i^t\text{ is finite and }a_i^t\leq\tau_t,\\
0, & \text{otherwise}.
\end{cases}
$$

Here, $b_i^t=1$ indicates that client $i$ is retained. If no client passes the threshold under extreme numerical conditions, the implementation retains the client with the smallest finite score, ensuring deterministic fallback behavior in the current round. In Phase I, $a_i^t=e_i^t$; in Phase II, $a_i^t$ is the fused score of standardized reconstruction error and SVDD distance. Therefore, the same MAD rule can accommodate different anomaly evidence across both phases without requiring server data, prior knowledge of the malicious-client ratio, or knowledge of the attack type.

Finally, the server performs hard isolation and equal-weight aggregation over retained clients:

$$
\alpha_i^t=\frac{b_i^t}{\sum_{j=1}^{K}b_j^t},\qquad
\Theta^{t}=\sum_{i=1}^{K}\alpha_i^t\Theta_i^t.
$$

Rejected clients are assigned strictly zero weight and do not participate in the global-state update for the current round. This design tightly couples anomaly detection with aggregation: the MAD threshold adaptively determines the trusted client set from the score distribution of the current round, AE-SVDD provides anomaly evidence that is less dependent on specific attack strategies, and the final equal-weight aggregation prevents highly anomalous clients from retaining influence in parameter averaging.