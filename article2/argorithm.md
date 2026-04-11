## Algorithm 1: AE-SVDD Federated Learning Robust Aggregation

Input: 
  Total communication rounds $T$,
  AE phase 1 warmup rounds $T_{warmup}$,
  SVDD threshold annealing rounds $T_{svdd}$,
  Total number of clients $K$,
  Threshold annealing parameters $\tau_{start}, \tau_{end}$,
  Center EMA decay rate $\gamma$,
  Reconstruction loss weight $\lambda$

Output:
  Global model $W_T$

1: Initialize global model $W_0$, AutoEncoder AE(Enc, Dec), Replay Buffer B, center c = None
2: for t = 1 to T do
3:     // --- Client Local Training ---
4:     for client k = 1 to K in parallel do
5:         Receive W_{t-1} from server, train on local data, and upload W_{t, k}
6:     end for
7:  
8:     // --- Server Aggregation ---
9:     Extract BN features X_{t, k} from each W_{t, k} and stack into matrix X_t
10:    Scale X_t using robust z-score: X_t = (X_t - Median(X_t)) / MAD(X_t)
11:    Add X_t to Replay Buffer B
12:
13:    if t <= T_warmup then  // Phase 1: AE Warm-up
14:        Sample batch X_buf from B
15:        Compute sample-wise L1 loss L = ||X_buf - Dec(Enc(X_buf))||*1
16:        // Trimmed reconstruction loss (keep samples <= 80th percentile)
17:        Loss_AE = Mean(L[L <= quantile(L, 0.8)])
18:        Update AE parameters by minimizing Loss_AE
19:        // Uniform aggregation
20:        W_t = (1 / K) * sum*{k=1}^K W_{t, k}
21:
22:    else  // Phase 2: SVDD Filtering
23:        Z_t = Enc(X_t)  // Get latent embeddings (no gradient)
24:  
25:        if c is None then  // Lazy Center Initialization
26:            Compute errors e = ||X_t - Dec(Z_t)||*1
27:            Mask_init = (e <= Median(e))
28:            c = Mean(Z_t[Mask_init])
29:        end if
30:*  
*31:        // Calculate SVDD distances
32:        d_k = ||Z*{t, k} - c||*2^2  for k = 1 to K
33:*  
*34:        // Adaptive Thresholding with Annealing tau
35:        p = min(1.0, (t - T_warmup) / T_svdd)
36:        tau_t = tau_start - p * (tau_start - tau_end)
37:        theta_t = Median(d) + tau_t * MAD(d)
38:*  
*39:        // Client Filtering
40:        TrustedSet S = {k | d_k <= theta_t}
41:        if S is empty then S = {1, ..., K} // Fallback
42:*  
*43:        // Weighted Aggregation
44:        W_t = (1 / |S|) * sum*{k in S} W_{t, k}
45:  
46:        // Center EMA Update
47:        c_new = Mean(Z_{t, k}) for k in S
48:        c = gamma * c + (1 - gamma) * c_new
49:  
50:        // Joint Fine-tuning
51:        Sample batch X_buf from B
52:        Loss_recon = Trimmed_L1(X_buf, Dec(Enc(X_buf)), keep=80%)
53:        Loss_svdd = (1 / |S|) * sum_{k in S} ||Enc(X_{t, k}) - c||_2^2  // with frozen Decoder
54:        Loss_total = Loss_svdd + lambda * Loss_recon
55:        Update AE parameters by minimizing Loss_total
56:    end if
57: end for