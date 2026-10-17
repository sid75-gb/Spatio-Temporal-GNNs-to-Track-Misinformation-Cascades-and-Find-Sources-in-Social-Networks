# Spatio-Temporal Graph Neural Networks for Misinformation Cascade Tracking and Source Node Identification in Social Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![PyG 2.5+](https://img.shields.io/badge/PyG-2.5+-3C2179.svg)](https://pyg.org/)
[![Status: Research Prototype](https://img.shields.io/badge/Status-Research%20Prototype-success.svg)](#)

>
> **Abstract:**
> The diffusion of misinformation across online social platforms presents significant risks to public discourse, democratic processes, and public health. Classical detection approaches rely primarily on natural language processing (NLP) of post text, leaving them vulnerable to adversarial paraphrasing, stylistic variation, and context deprivation. This project investigates structural and temporal representations of information diffusion by modeling misinformation propagation as dynamic, timestamp-ordered cascade trees on Twitter interaction graphs.
>
> Using the User Preference-aware Fake News Detection (**UPFD**) benchmark across political (**PolitiFact**) and entertainment (**GossipCop**) domains, we implement a **Spatio-Temporal Graph Neural Network (STGNN)** combining 778-dimensional multimodal node representations (768-d BERT text embeddings fused with 10-d user profile attributes), bidirectional spatial Graph Attention (GAT) layers, temporal Gated Recurrent Units (GRU), multi-head temporal self-attention, and global attention graph pooling. The primary evaluated task is **graph-level binary veracity classification (`REAL` vs. `FAKE`)**.
>
> Furthermore, the framework evaluates **early-stage detection capability** across chronological observation milestones (25%, 50%, 75%, and 100% user arrival quantiles) on held-out test cascades, provides a local **interactive offline visual analytics dashboard** for cascade exploration, and conducts a separate **exploratory source-proxy analysis** evaluating structural ranking heuristics against earliest-observed retweeter timestamps.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Current Implementation](#2-current-implementation)
3. [Dataset Details & Splits](#3-dataset-details--splits)
4. [Methodology](#4-methodology)
5. [Experimental Results](#5-experimental-results)
   - [A. Early-Stage Milestone Evaluation](#a-early-stage-milestone-evaluation)
   - [B. Extended-Epoch Checkpoint Evaluation](#b-extended-epoch-checkpoint-evaluation)
   - [C. Standardized 10-Epoch Baseline Comparison](#c-standardized-10-epoch-baseline-comparison)
6. [Exploratory Source-Proxy Experiment](#6-exploratory-source-proxy-experiment)
7. [Interactive Cascade Dashboard](#7-interactive-cascade-dashboard)
8. [Installation & Usage](#8-installation--usage)
9. [Repository Structure](#9-repository-structure)
10. [Current Limitations & Planned Future Work](#10-current-limitations--planned-future-work)
11. [References](#11-references)

---

## 1. Project Overview

### Research Context & Objective
Misinformation diffusion is intrinsically a socio-technical process governed both by *content* (what claim is made) and by *topology and dynamics* (how, when, and through whom the claim diffuses). While pure linguistic classifiers evaluate the root article or individual retweets in isolation, cascade trees provide rich structural signals:
- Misinformation cascades frequently exhibit distinctive structural virality, deeper propagation trees, and bursty retweeting dynamics.
- Graph neural networks can leverage propagation structure to distinguish organic news spreading from coordinated amplification patterns.

### Scope of Current Implementation
The core, empirically verified objective of this repository is **graph-level binary classification** of social media cascades as either `REAL` (label 0) or `FAKE` (label 1).

Key aspects of the current scope include:
- **Primary Evaluated Task:** Supervised graph-level veracity classification evaluated on official held-out test splits of the UPFD benchmark.
- **Early-Stage Evaluation:** Systematic evaluation of detection performance when observing only early chronological fractions of cascades (25%, 50%, 75%, and 100% user arrival quantiles).
- **Exploratory Source Analysis:** Heuristic structural ranking (Shah & Zaman Rumor Centrality and tree out-degree) evaluated against an observational proxy--the *earliest-observed retweeter*.
- **Offline Visualization Dashboard:** A local, web-based visual analytics interface for exploring cascade diffusion trees, temporal snapshots, and prediction metrics.

> [!IMPORTANT]
> **Scientific Scoping & Ground Truth Notice:**
> The UPFD benchmark does **not** contain verified human author labels, ground-truth seed account annotations, or verified bot-account labels. Node 0 in every UPFD cascade represents the news article claim, while nodes 1 to $N-1$ represent retweeting user accounts. Consequently, source-localization experiments in this repository evaluate an observational timestamp proxy (the earliest retweeter in the crawl), and the STGNN source-scoring head is not trained under a supervised source loss.

---

## 2. Current Implementation

The model architecture combines multimodal feature fusion, bidirectional spatial graph attention, recurrent temporal modeling, and attention pooling into a unified end-to-end framework:

```
                      [News Post Root (Node 0) + Retweet User Nodes (1..N-1)]
                                                 |
                     +---------------------------+---------------------------+
                     |                                                       |
                     v                                                       v
           [768-d BERT Text Vector]                                [10-d User Profile Vector]
         (Root news text & tweet text)                           (Followers, friends, verified, age)
                     |                                                       |
                     +---------------------------+---------------------------+
                                                 |
                                                 v
                                     [778-d Node Feature Matrix]
                                                 |
                                                 v
                                  [Temporal Cascade Builder]
                      Chronological snapshots: G^(25%) --> G^(50%) --> G^(75%) --> G^(100%)
                      (Optional 4 dynamic structural features: depth, degree, progress, active)
                                                 |
                     +---------------------------+---------------------------+
                     |                                                       |
                     v                                                       v
            [Forward GAT Conv]                                      [Reverse GAT Conv]
           (Parent --> Child flow)                                 (Child --> Parent flow)
                     |                                                       |
                     +---------------------------+---------------------------+
                                                 |
                                                 v
                                   [Bidirectional Feature Fusion]
                                          LayerNorm + GELU
                                                 |
                                                 v
                                      [Temporal Recurrent Cell]
                                        (Node-level GRU Cell)
                                                 |
                                                 v
                                    [Temporal Multi-Head Attention]
                                      (Trajectory self-attention)
                                                 |
                                                 v
                                     [Final Node Embeddings Z]
                                                 |
                     +---------------------------+---------------------------+
                     |                                                       |
                     v                                                       v
         [Global Attention Pooling]                                [Root Node Embedding Z_0]
         (Softmax gate across nodes)                              (Original claim representation)
                     |                                                       |
                     +---------------------------+---------------------------+
                                                 |
                                                 v
                                    [Concatenated Graph Readout]
                                                 |
                                                 v
                                  [MLP Binary Veracity Classifier]
                                     Logits --> [REAL vs. FAKE]
```

### Component Details
1. **Multimodal Node Representations:**
   - 768-dimensional contextual BERT embeddings representing the news article text (for root node 0) and retweet content.
   - 10-dimensional social profile features capturing user-level attributes: follower count, friend count, verified status, account age, and status counts.
   - Fused input representation: **778 dimensions** per node. Normalization statistics (mean and variance) are computed exclusively on the training split to eliminate data leakage.
2. **Chronological Snapshot Generation:**
   - Evaluates diffusion over 4 observation quantiles: 25%, 50%, 75%, and 100% of observed retweeters.
   - Strictly enforces causal non-leakage: future nodes have zeroed features and future edges are excluded from early snapshots.
   - Optionally appends 4 dynamic temporal-structural features (normalized cascade depth, normalized out-degree, elapsed observation fraction, and active indicator), resulting in a 782-dimensional representation at each snapshot.
3. **Bidirectional Spatial GAT Layers:**
   - Forward attention computes message passing along the direction of information contagion (parent $\to$ child).
   - Reverse attention aggregates structural feedback from children to parents (child $\to$ parent).
   - Fused with layer normalization and dropout ($p=0.3$).
4. **Temporal Recurrent & Attention Encoding:**
   - Recurrent update via node-level Gated Recurrent Unit (`GRUCell`, hidden dimension 64).
   - Multi-head temporal self-attention across the snapshot trajectory with key padding masks for inactive steps.
5. **Graph-Level Readout & Classification:**
   - Global attention pooling (`GlobalAttention`) aggregates node states into a cascade-level vector.
   - Concatenated with the root node embedding ($Z_0$) to preserve the core claim's representation.
   - Multi-layer perceptron (Linear $\to$ LayerNorm $\to$ GELU $\to$ Dropout $\to$ Linear) outputs logits for binary cross-entropy optimization.
6. **Auxiliary Driver-Scoring Head (Untrained):**
   - A linear head mapping node embeddings to scalar driver scores. As documented in `src/models/stgnn.py`, this head is not trained under a supervised loss due to lack of ground-truth author labels.
7. **Heuristic Driver & Amplifier Indicators:**
   - Structural propagation-driver candidates are ranked via Shah & Zaman Rumor Centrality.
   - Potential amplification hubs are flagged using heuristic tree fan-out and burst velocity Z-score thresholds. These serve as exploratory visual indicators rather than verified account classifications.

---

## 3. Dataset Details & Splits

Evaluations are conducted on the **User Preference-aware Fake News Detection (UPFD)** benchmark ([Dou et al., SIGIR 2021](https://dl.acm.org/doi/10.1145/3404835.3462990)):

### Verified Graph & Split Counts

| Dataset | Total Cascades | Train Split | Validation Split | Test Split (Held-Out) | Primary Topic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **UPFD PolitiFact** | 314 | 62 | 31 | 221 | Political news and fact-checking statements |
| **UPFD GossipCop** | 5,464 | 1,092 | 546 | 3,826 | Celebrity rumors and entertainment stories |

### Graph Semantics & Modalities
- **Graph Labels ($y \in \{0, 1\}$):** Binary ground truth where `0 = REAL` and `1 = FAKE`.
- **Node Semantics:**
  - **Node 0:** Represents the original news article / publisher claim. Node 0 does not possess a Twitter retweet arrival timestamp in UPFD.
  - **Nodes $1 \dots N-1$:** Represent user accounts that retweeted or propagated the story.
- **Edge Direction:** Directed propagation edges $(u, v)$ represent information flow from spreader $u$ to retweeter $v$.
- **Timestamp Coverage:** Real retweet arrival timestamps (Unix epoch seconds) are loaded from official mapping files. In the held-out test splits, all 221 PolitiFact cascades and all 3,826 GossipCop cascades have complete timestamp coverage across their propagating user nodes.

### Local Data & Checkpoint Artifacts
Due to licensing and file size limitations, raw datasets and model checkpoints are **not committed to the GitHub repository** (enforced via `.gitignore`).

To run training or evaluation locally, the following files are required in `data/UPFD/`:
```
data/UPFD/
|-- politifact/
|   \-- raw/
|       |-- A.txt
|       |-- graph_labels.npy
|       |-- node_graph_id.npy
|       |-- train_idx.npy / val_idx.npy / test_idx.npy
|       |-- new_bert_feature.npz
|       |-- new_profile_feature.npz
|       \-- pol_id_time_mapping.pkl       # Official timestamp mapping
\-- gossipcop/
    \-- raw/
        |-- A.txt
        |-- graph_labels.npy
        |-- node_graph_id.npy
        |-- train_idx.npy / val_idx.npy / test_idx.npy
        |-- new_bert_feature.npz
        |-- new_profile_feature.npz
        \-- gos_id_time_mapping.pkl       # Official timestamp mapping
```

> [!NOTE]
> The official UPFD raw graph files and `<prefix>_id_time_mapping.pkl` dictionary files are distributed by the original UPFD authors. See [Dou et al. (2021)](https://github.com/safe-graph/GNN-FakeNews) and PyG's `torch_geometric.datasets.UPFD` documentation.

---

## 4. Methodology

### End-to-End Processing Pipeline

1. **Dataset Ingestion & Split Integrity:**
   - Graphs are loaded via PyTorch Geometric's `UPFD` interface using official 20%/10%/70% train/validation/test splits.
   - Text features (768-d BERT) and social profiles (10-d) are fused into a 778-d vector per node.
   - Normalization statistics for profile features are computed exclusively on the training split:
     $$\mu_{\text{train}} = \frac{1}{N_{\text{train}}} \sum x, \quad \sigma_{\text{train}} = \sqrt{\frac{1}{N_{\text{train}}} \sum (x - \mu_{\text{train}})^2 + \epsilon}$$
2. **Chronological Snapshot Construction:**
   - For each cascade, user nodes ($v \ge 1$) are sorted by their observed retweet timestamp.
   - Snapshots $G^{(1)}, G^{(2)}, G^{(3)}, G^{(4)}$ correspond to chronological observation fractions:
     $$\mathcal{F} = \{0.25, 0.50, 0.75, 1.00\}$$
   - Cutoff timestamps are selected as quantiles of the observed user population.
   - Nodes appearing strictly after the cutoff are masked out: their node features are set to zero, and any connected future edges are removed from the snapshot adjacency matrix.
   - Root node 0 is active across all snapshots as the originating news story.
3. **Spatio-Temporal Representation Learning:**
   - At each snapshot $t$, forward and backward GAT layers perform directional message passing over active subgraphs.
   - Recurrent GRU cells update temporal node memory across successive snapshots.
   - Multi-head temporal self-attention models trajectory dynamics across observation stages.
4. **Graph Readout & Loss Optimization:**
   - Graph representations are formed by concatenating attention-pooled node embeddings with the root post embedding.
   - The model is optimized using cross-entropy loss with AdamW ($\text{lr}=0.001$, $\text{weight decay}=10^{-4}$), gradient clipping (max norm 5.0), and learning rate reduction on plateau.
   - Supervised loss is computed exclusively on graph-level veracity labels.

---

## 5. Experimental Results

All reported metrics are verified directly from saved evaluation artifacts in `outputs/`. Metrics are reported on the official held-out test splits (PolitiFact: $N=221$; GossipCop: $N=3,826$).

### A. Early-Stage Milestone Evaluation

Evaluates whether misinformation cascades can be detected early in their lifecycle before full saturation. Snapshots correspond to chronological user arrival quantiles (variable-length prefix evaluation using `model.num_snapshots=0` with trained timestamp checkpoints):

*Source artifacts: [`outputs/early_stage_politifact_stgnn.json`](file:///c:/Users/Sid/Desktop/GNN/outputs/early_stage_politifact_stgnn.json) and [`outputs/early_stage_gossipcop_stgnn.json`](file:///c:/Users/Sid/Desktop/GNN/outputs/early_stage_gossipcop_stgnn.json)*

#### UPFD PolitiFact (Test Set: $N=221$ Cascades; Real: 108, Fake: 113)

| Observation Stage | Prefix Snapshots ($T$) | Accuracy | Macro-F1 | Binary-F1 | Macro-Precision | Macro-Recall | ROC-AUC | TP | FP | TN | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **25% Observation** | 1 | 0.8416 | 0.8416 | 0.8402 | 0.8427 | 0.8423 | 0.8795 | 92 | 14 | 94 | 21 |
| **50% Observation** | 2 | 0.8371 | 0.8371 | 0.8349 | 0.8385 | 0.8378 | 0.8794 | 91 | 14 | 94 | 22 |
| **75% Observation** | 3 | 0.8416 | 0.8416 | 0.8402 | 0.8427 | 0.8423 | 0.8801 | 92 | 14 | 94 | 21 |
| **100% Full Cascade** | 4 | 0.8371 | 0.8371 | 0.8364 | 0.8378 | 0.8376 | 0.8818 | 92 | 15 | 93 | 21 |

#### UPFD GossipCop (Test Set: $N=3,826$ Cascades; Real: 1,910, Fake: 1,916)

| Observation Stage | Prefix Snapshots ($T$) | Accuracy | Macro-F1 | Binary-F1 | Macro-Precision | Macro-Recall | ROC-AUC | TP | FP | TN | FN |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **25% Observation** | 1 | 0.9320 | 0.9320 | 0.9306 | 0.9329 | 0.9321 | 0.9537 | 1743 | 87 | 1823 | 173 |
| **50% Observation** | 2 | 0.9467 | 0.9467 | 0.9456 | 0.9474 | 0.9467 | 0.9651 | 1774 | 62 | 1848 | 142 |
| **75% Observation** | 3 | 0.9577 | 0.9577 | 0.9571 | 0.9581 | 0.9577 | 0.9763 | 1806 | 52 | 1858 | 110 |
| **100% Full Cascade** | 4 | 0.9660 | 0.9660 | 0.9659 | 0.9660 | 0.9660 | 0.9837 | 1843 | 57 | 1853 | 73 |

*Key Insight:* On GossipCop, performance scales monotonically as more temporal diffusion context becomes available (Accuracy increases from 93.20% at 25% observation to 96.60% at full observation, with ROC-AUC rising from 0.9537 to 0.9837). On PolitiFact, strong early detection (84.16% Accuracy at 25% observation) is maintained across all stages, reflecting strong early topological separation in political cascades.

---

### B. Extended-Epoch Checkpoint Evaluation

Evaluates the best saved timestamp-trained STGNN model checkpoints saved under early stopping (epoch 20, learning rate 0.001, early stopping patience 8, seed 42):

*Source artifacts: [`outputs/stgnn_politifact_timestamp_lr001_es8.json`](file:///c:/Users/Sid/Desktop/GNN/outputs/stgnn_politifact_timestamp_lr001_es8.json) and [`outputs/stgnn_gossipcop_timestamp_lr001_es8.json`](file:///c:/Users/Sid/Desktop/GNN/outputs/stgnn_gossipcop_timestamp_lr001_es8.json)*

| Dataset | Test Size ($N$) | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Binary-F1 | ROC-AUC | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PolitiFact** | 221 | **0.8371** | 0.8378 | 0.8376 | **0.8371** | 0.8364 | **0.8818** | 0.5544 |
| **GossipCop** | 3,826 | **0.9660** | 0.9660 | 0.9660 | **0.9660** | 0.9659 | **0.9837** | 0.1501 |

---

### C. Standardized 10-Epoch Baseline Comparison

To ensure fair comparison across architectural paradigms, a separate standardized benchmark was executed where all models were trained for exactly **10 epochs** under identical data splits, learning rate (0.001), batch size (16), and random seed (42):
- **BERT NLP Baseline:** Content-only MLP operating strictly on the root news post text embedding (`src/models/nlp_baseline.py`).
- **Static GAT Baseline:** Static 2-layer Graph Attention Network with attention pooling operating on the full propagation tree without temporal decomposition (`src/models/static_gnn.py`).
- **STGNN (Ours):** Spatio-Temporal GNN with 4 timestamp snapshots (`src/models/stgnn.py`).

*Source artifacts: [`outputs/benchmark_politifact.json`](file:///c:/Users/Sid/Desktop/GNN/outputs/benchmark_politifact.json) and [`outputs/benchmark_gossipcop.json`](file:///c:/Users/Sid/Desktop/GNN/outputs/benchmark_gossipcop.json)*

#### UPFD PolitiFact Benchmark (10 Epochs, Held-Out Test $N=221$)

| Model Architecture | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | ROC-AUC | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Linguistic Content (BERT NLP)** | 0.7828 | 0.7849 | 0.7837 | 0.7827 | 0.8441 | 0.5068 |
| **2. Static Cascade GNN (Static GAT)** | 0.7602 | 0.7763 | 0.7628 | 0.7578 | 0.8522 | 0.5630 |
| **3. Proposed STGNN (Ours)** | **0.8054** | **0.8123** | **0.8071** | **0.8049** | **0.8550** | **0.4713** |

*PolitiFact Finding:* At 10 epochs, STGNN outperforms both the linguistic-only baseline (+2.22% F1) and the static GNN (+4.71% F1), validating the utility of temporal ordering in smaller political cascades.

#### UPFD GossipCop Benchmark (10 Epochs, Held-Out Test $N=3,826$)

| Model Architecture | Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | ROC-AUC | Test Loss |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Linguistic Content (BERT NLP)** | 0.7297 | 0.7382 | 0.7299 | 0.7274 | 0.8056 | 0.5394 |
| **2. Static Cascade GNN (Static GAT)** | **0.9650** | **0.9653** | **0.9650** | **0.9650** | **0.9918** | **0.1308** |
| **3. Proposed STGNN (Ours)** | 0.9587 | 0.9588 | 0.9587 | 0.9587 | 0.9864 | 0.1404 |

*GossipCop Finding & Empirical Honesty:*
In this standardized 10-epoch regime on GossipCop, the Static GAT model achieves higher accuracy (0.9650 vs. 0.9587) and higher ROC-AUC (0.9918 vs. 0.9864) than the 10-epoch STGNN. Graph structure provides a massive boost over pure text (+23.76% F1 for Static GAT over BERT NLP), but the static model learns faster per epoch on large cascades due to lower parameter complexity. When trained for 20 epochs with early stopping (Section B), STGNN reaches 0.9660 Accuracy and 0.9837 ROC-AUC.

---

## 6. Exploratory Source-Proxy Experiment

### Problem Formulation & Target Definition
In social network diffusion research, identifying the origin of misinformation is of major interest. However, **UPFD does not provide verified human author labels or account-level source ground truth**. Node 0 in UPFD represents the news article claim.

To evaluate source identification objectively without making unsupported scientific claims, we define an observational proxy target:
- **Proxy Target:** The **earliest-observed retweeter** among user accounts ($v \ge 1$) having a valid, finite timestamp.
- **Candidate Pool:** All active user nodes in the cascade ($1 \dots N-1$). Root node 0 is strictly excluded from candidate ranking.
- **Tie Handling:** Multi-target convention (a prediction is credited as a hit if the top-$K$ candidates contain any tied earliest retweeter; the best rank is used for Mean Reciprocal Rank).

### Methods Evaluated
1. **Random User Baseline:** Selects a candidate user uniformly at random (evaluated with seed 42, alongside analytical expectation $E[\text{Hit@}K] \approx K/M$).
2. **Out-Degree Centrality:** Ranks users by out-degree in the directed cascade tree (highest retweet fan-out).
3. **Rumor Centrality ([Shah & Zaman, 2011](https://ieeexplore.ieee.org/document/5961803)):** Structural message-passing algorithm counting the number of permitted linear extensions of the diffusion tree rooted at candidate node $v$:
   $$R(v, G) = N! \prod_{u \in V} \frac{1}{|T_u(v)|}$$
4. **Sanity-Check Oracle:** Sorts candidate users directly by their observed timestamp. This is an **upper-bound sanity check** confirming metric pipeline correctness; it is **not** a discovery method.

*Source artifacts: [`outputs/source_proxy_politifact.json`](file:///c:/Users/Sid/Desktop/GNN/outputs/source_proxy_politifact.json) and [`outputs/source_proxy_gossipcop.json`](file:///c:/Users/Sid/Desktop/GNN/outputs/source_proxy_gossipcop.json)*

### Experimental Results on Held-Out Test Splits

| Dataset | Method / Ranking Strategy | Hit@1 | Hit@3 | Hit@5 | Mean Reciprocal Rank (MRR) | Mean Hop Distance |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PolitiFact**<br>($N=221$ test cascades,<br>6 tied [2.71%]) | Random Baseline (Seed 42) | 0.0588 | 0.1041 | 0.1403 | 0.1216 | 2.35 |
| | Random Analytical Expectation | 0.0451 | 0.1205 | 0.1725 | 0.1190 | -- |
| | Out-Degree Centrality | 0.0995 | 0.1493 | 0.2127 | 0.1674 | 1.86 |
| | **Rumor Centrality (Shah & Zaman)** | **0.0995** | **0.1584** | **0.2308** | **0.1805** | **1.81** |
| | *[Oracle] Earliest Timestamp (Upper Bound)* | *1.0000* | *1.0000* | *1.0000* | *1.0000* | *0.00* |
| **GossipCop**<br>($N=3,826$ test cascades,<br>62 tied [1.62%]) | Random Baseline (Seed 42) | 0.0452 | 0.1422 | 0.2256 | 0.1466 | 2.07 |
| | Random Analytical Expectation | 0.0490 | 0.1438 | 0.2259 | 0.1488 | -- |
| | Out-Degree Centrality | 0.1357 | 0.2352 | 0.3048 | 0.2311 | 1.76 |
| | **Rumor Centrality (Shah & Zaman)** | **0.1390** | **0.2480** | **0.3327** | **0.2429** | **1.72** |
| | *[Oracle] Earliest Timestamp (Upper Bound)* | *1.0000* | *1.0000* | *1.0000* | *1.0000* | *0.00* |

### Scientific Conclusions on Source Localization
- **Rumor Centrality achieves modest predictive utility:** Rumor Centrality outperforms random guessing (+4.07% Hit@1 on PolitiFact; +9.38% Hit@1 on GossipCop) and achieves lower topological error distance (1.81 vs. 2.35 hops on PolitiFact; 1.72 vs. 2.07 hops on GossipCop).
- **Realistic Heuristic Performance:** Structural heuristics achieve modest accuracy (approximately 10% to 14% Hit@1) against the earliest retweeter proxy. The 1.0000 Hit@1 score reflects only the timestamp oracle upper-bound sanity check, not a predictive discovery model.
- **Untrained STGNN Head:** The STGNN's `source_head` is untrained because UPFD lacks supervised author labels. It is not presented as a validated source detector.

---

## 7. Interactive Cascade Dashboard

The repository includes a web-based visual analytics dashboard for inspecting misinformation cascades and understanding spatio-temporal dynamics:

```
web/
|-- index.html                # Visual cascade analytics interface (D3 / Vis.js)
|-- dashboard_data.json       # Precomputed test cascade records (PolitiFact + GossipCop)
\-- vis-network.min.js        # Standalone network visualization library
```

### Supported Functionality
- **Dataset Switching:** Toggle between UPFD PolitiFact and UPFD GossipCop test cascades.
- **Cascade Selector:** Browse individual cascades by ID, true label (`REAL` vs. `FAKE`), node count, max tree depth, and structural virality.
- **Chronological Quantile Progression:** Step through temporal snapshots ($T_1 \to T_4$ representing 25%, 50%, 75%, and 100% observation) to visualize how the tree expands over time.
- **Prediction Panel:** Displays model veracity prediction, class probabilities, and correctness indicator at each temporal phase.
- **Node Role Inspection:** Interactive Vis.js network graph highlighting:
  - **Root News Post (Blue):** Node 0 representing the claim.
  - **Propagation Driver Candidates (Gold):** Ranked by Shah & Zaman Rumor Centrality.
  - **Propagation Amplifiers (Red):** Identified via heuristic fan-out and burst velocity thresholds.
  - **Standard Retweeters (Slate):** Downstream retweeting users.
- **Benchmark Summary Tables:** Integrated presentation of 10-epoch baselines, extended-epoch checkpoints, and early-stage evaluation results.

> [!NOTE]
> **Operational Boundary:**
> The dashboard is an **offline research visualization tool** for inspecting benchmark cascades. It is not a live social-media monitoring service, and its structural heuristic markers (driver candidates and amplifier nodes) are exploratory indicators rather than verified real-world account or bot predictions.

---

## 8. Installation & Usage

### Prerequisites
- **Operating System:** Windows 10/11 (PowerShell), Linux, or macOS.
- **Python:** Version 3.10 or 3.11 recommended.
- **GPU (Optional):** CUDA-compatible GPU (defaults to CPU execution).

### 1. Environment Setup (Windows PowerShell)

```powershell
# Clone the repository
git clone https://github.com/sid75-gb/Spatio-Temporal-GNNs-to-Track-Misinformation-Cascades-and-Find-Sources-in-Social-Networks.git
cd Spatio-Temporal-GNNs-to-Track-Misinformation-Cascades-and-Find-Sources-in-Social-Networks

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Dataset Setup
As explained in [Section 3](#3-dataset-details--splits), raw UPFD graph data and timestamp mapping files are not committed to GitHub.
1. Download or generate the UPFD files for `politifact` and `gossipcop`.
2. Ensure auxiliary mapping files are placed in `data/UPFD/<dataset>/raw/`:
   - `pol_id_time_mapping.pkl` (for PolitiFact)
   - `gos_id_time_mapping.pkl` (for GossipCop)
   - `node_graph_id.npy`, `train_idx.npy`, `val_idx.npy`, `test_idx.npy`

### 3. Launching the Interactive Dashboard

If `web/dashboard_data.json` is already present (committed in the repository):
```powershell
# Option A: Start Python's built-in web server directly
python -m http.server 8080 --directory web
```
Then navigate in your browser to `http://localhost:8080`.

Alternatively, regenerate fresh dashboard data from saved checkpoints:
```powershell
# Option B: Run the dashboard server runner (requires local checkpoints & data)
python scripts/demo_dashboard.py --port 8080 --open --dataset all
```

### 4. Running Early-Stage Evaluation
To evaluate early-stage detection performance across 25%, 50%, 75%, and 100% observation on held-out test splits (requires local data and checkpoints):
```powershell
# Evaluate both PolitiFact and GossipCop
python scripts/evaluate_early_stage.py --dataset all --device cpu

# Evaluate PolitiFact only
python scripts/evaluate_early_stage.py --dataset politifact --device cpu
```
Results are saved to `outputs/early_stage_politifact_stgnn.json` and `outputs/early_stage_gossipcop_stgnn.json`.

### 5. Running the Source-Proxy Experiment
To evaluate the heuristic ranking algorithms against the earliest-observed retweeter proxy:
```powershell
python scripts/evaluate_source_proxy.py --dataset all --seed 42
```
Results are saved to `outputs/source_proxy_politifact.json` and `outputs/source_proxy_gossipcop.json`.

### 6. Training Models (Optional)
To retrain models from scratch using official timestamp mappings:
```powershell
# Train STGNN on PolitiFact with timestamps
python scripts/train.py --model stgnn --dataset politifact --epochs 20 --lr 0.001 --early_stopping_patience 8 --temporal_mode timestamp --multimodal

# Train Static GAT baseline
python scripts/train.py --model static_gnn --dataset politifact --epochs 10 --lr 0.001 --multimodal

# Train BERT NLP baseline
python scripts/train.py --model nlp --dataset politifact --epochs 10 --lr 0.001
```

### 7. Automated Test Suite & Known Status
The repository includes unit tests in `tests/test_pipeline.py`.

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

> [!WARNING]
> **Test Suite Notice:**
> The current `tests/test_pipeline.py` suite contains legacy tests created prior to the timestamp-integrity refactoring. Because `TemporalCascadeBuilder` now strictly requires real timestamp metadata (refusing to invent synthetic timestamps), and `source_identifier.py` explicitly rejects `true_source=0` (enforcing that news root 0 is not a user source), running the unit test file directly results in 3 expected failures. Tests covering cascade feature computation, static GNN, NLP baseline, rumor centrality, and bot isolation pass successfully.

---

## 9. Repository Structure

```
GNN/
|-- configs/
|   \-- config.yaml                     # Model hyperparameters & training configurations
|-- src/
|   |-- __init__.py
|   |-- algorithms/
|   |   |-- __init__.py
|   |   |-- bot_detection.py            # Heuristic fan-out & velocity anomaly scoring
|   |   \-- rumor_centrality.py         # Shah & Zaman message passing & Jordan center
|   |-- data/
|   |   |-- __init__.py
|   |   |-- cascade_features.py         # Structural virality (Wiener index), depth, degrees
|   |   |-- dataset_loader.py           # UPFD loader, profile fusion, timestamp resolver
|   |   \-- temporal_builder.py         # Chronological quantile snapshot builder
|   |-- evaluation/
|   |   |-- __init__.py
|   |   \-- metrics.py                  # Classification and ranking evaluation metrics
|   |-- models/
|   |   |-- __init__.py
|   |   |-- nlp_baseline.py             # Root text content MLP classifier
|   |   |-- source_identifier.py        # Structural candidate ranker (excluding node 0)
|   |   |-- static_gnn.py               # Static GAT / GCN / GraphSAGE baselines
|   |   \-- stgnn.py                    # Spatio-Temporal GNN (Bidirectional GAT + GRU)
|   \-- visualization/
|       |-- __init__.py
|       \-- cascade_visualizer.py       # Static cascade visualization utilities
|-- scripts/
|   |-- demo_dashboard.py               # Dashboard data generator & local HTTP server
|   |-- evaluate.py                     # Standardized 10-epoch comparative benchmark
|   |-- evaluate_early_stage.py         # Variable-length early-stage milestone evaluation
|   |-- evaluate_source_proxy.py        # Earliest-observed retweeter proxy evaluation
|   |-- identify_sources.py             # Propagation driver analysis demo script
|   \-- train.py                        # Model training pipeline with early stopping
|-- web/
|   |-- index.html                      # Interactive Vis.js cascade analytics dashboard
|   |-- dashboard_data.json             # Precomputed test evaluation dataset for dashboard
|   \-- vis-network.min.js              # Visualization library bundle
|-- tests/
|   \-- test_pipeline.py                # Pipeline unit & integration tests
|-- outputs/                            # Verified evaluation result artifacts (.json)
|   |-- benchmark_politifact.json       # 10-epoch standardized benchmark (PolitiFact)
|   |-- benchmark_gossipcop.json        # 10-epoch standardized benchmark (GossipCop)
|   |-- early_stage_politifact_stgnn.json  # 25/50/75/100% early-stage evaluation
|   |-- early_stage_gossipcop_stgnn.json   # 25/50/75/100% early-stage evaluation
|   |-- source_proxy_politifact.json    # PolitiFact proxy evaluation results
|   |-- source_proxy_gossipcop.json     # GossipCop proxy evaluation results
|   |-- stgnn_politifact_timestamp_lr001_es8.json  # Extended checkpoint evaluation
|   \-- stgnn_gossipcop_timestamp_lr001_es8.json   # Extended checkpoint evaluation
|-- requirements.txt                    # Project Python package dependencies
|-- .gitignore                          # Excludes large data, checkpoints, and scratch files
\-- README.md                           # Research documentation and user guide
```

---

## 10. Current Limitations & Planned Future Work

To maintain research integrity, we explicitly delineate verified current capabilities from open limitations and planned research extensions:

### Current Limitations
1. **Observational Retweeter Proxy vs. True Misinformation Originators:**
   The UPFD benchmark lacks ground-truth annotations identifying the initial human creator or publisher accounts. The source-proxy experiment evaluates an observational heuristic (the earliest retweeter in the crawl), which may reflect platform crawl latency or missing historical tweets rather than true authorship.
2. **Untrained Auxiliary Source Head:**
   The node-level scoring head in `SpatioTemporalGNN` is not trained with a supervised objective. Real-world source identification requires dedicated ground-truth datasets with confirmed author labels.
3. **Discrete Quantile Snapshots vs. Continuous Streaming:**
   Snapshots represent discrete quantiles of total observed users (25%, 50%, 75%, 100%), rather than fixed real-time continuous streaming windows (e.g., 1 hour, 6 hours, 24 hours).
4. **Heuristic Propagation Indicators vs. Supervised Bot Detection:**
   The amplifier identification module relies on structural fan-out and burst velocity heuristics. It has not been trained or validated against external annotated bot benchmarks (such as Botometer or TwiBot-22).

### Planned Future Work (Not Implemented in Current Codebase)
- [ ] **Probability Calibration & Reliability Analysis:** Implement temperature scaling and Expected Calibration Error (ECE) analysis to quantify model prediction confidence under distribution shifts.
- [ ] **Explainable AI (XAI) for Cascades:** Integrate GNNExplainer and SubgraphX to identify which specific propagation sub-trees and user attributes drive veracity predictions.
- [ ] **Interactive Cascade Perturbation Simulations:** Enable dashboard users to simulate edge removals and node ablations to test cascade resilience against moderation interventions.
- [ ] **Validated New-Cascade Streaming Inference Workflow:** Develop end-to-end ingestion and real-time inference pipelines for previously unobserved live cascades.
- [ ] **Supervised Source-Node Learning:** Formulate source localization on synthetic or annotated diffusion benchmarks with verified origin ground truth.
- [ ] **Cross-Platform Domain Generalisation:** Validate models on social platforms beyond Twitter/X, such as Reddit (threaded discussions) and Telegram (forwarding channels).

---

## 11. References

1. **UPFD Benchmark:**
   Dou, Y., Shu, K., Xia, C., Yu, P. S., & Sun, L. (2021). *User Preference-aware Fake News Detection.* In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21), pp. 2051-2055. [doi:10.1145/3404835.3462990](https://doi.org/10.1145/3404835.3462990).
2. **Rumor Centrality & Source Detection:**
   Shah, D., & Zaman, T. (2011). *Rumors in a Network: Who's the Culprit?* IEEE Transactions on Information Theory, 57(8), pp. 5163-5181. [doi:10.1109/TIT.2011.2158885](https://doi.org/10.1109/TIT.2011.2158885).
3. **Structural Virality in Diffusion Networks:**
   Goel, S., Anderson, A., Hofman, J., & Watts, D. J. (2016). *The Structural Virality of Online Diffusion.* Management Science, 62(1), pp. 180-196. [doi:10.1287/mnsc.2015.2158](https://doi.org/10.1287/mnsc.2015.2158).
4. **Graph Attention Networks (GAT):**
   Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). *Graph Attention Networks.* International Conference on Learning Representations (ICLR '18). [arXiv:1710.10903](https://arxiv.org/abs/1710.10903).
5. **PyTorch Geometric (PyG):**
   Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.* ICLR Workshop on Representation Learning on Graphs and Manifolds.
