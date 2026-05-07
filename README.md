# Q-SNet: Quaternion Spiking Attention Network for Cross-Subject EEG Emotion Recognition

<p align="center">
  <img src="fig1.png" width="950">
</p>

<p align="center">
  <b>Q-SNet</b> is a quaternion-based spiking framework for cross-subject EEG emotion recognition.
  It combines quaternion multi-channel coupling, quaternion rotation attention, and magnitude-triggered quaternion LIF neurons.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Task-Cross--Subject%20EEG%20Emotion%20Recognition-blue">
  <img src="https://img.shields.io/badge/Model-Quaternion%20Spiking%20Network-green">
  <img src="https://img.shields.io/badge/Protocol-LOSOCV-orange">
  <img src="https://img.shields.io/badge/Datasets-SEED%20%7C%20SEED--IV%20%7C%20SEED--V-purple">
</p>

---

## Overview

Cross-subject EEG emotion recognition is difficult because EEG responses vary substantially across individuals. This induces distribution shifts in both temporal dynamics and inter-channel spatial patterns. Most existing models are built on real-valued representations and learn channel dependencies implicitly through convolution, graph propagation, or attention. Such representations may not preserve the structural relationships among EEG channels when transferred to unseen subjects.

Q-SNet addresses this issue by introducing quaternion algebra into spiking EEG modeling. Instead of treating EEG channels as independent scalar signals, Q-SNet groups spatially related EEG channels into quaternion representations, applies learnable quaternion rotations in the attention module, and triggers spikes according to quaternion magnitude. This design aims to improve cross-subject robustness while retaining the compact and event-driven characteristics of spiking neural networks.

---

## Core Idea

The central idea is simple:

> EEG channels are not independent measurements.  
> Q-SNet explicitly couples related EEG channels as quaternion components and performs attention, rotation, and spiking dynamics in quaternion space.

Q-SNet contains three main components:

1. **Quaternion-based DE feature extraction**  
   Region-aware EEG channel groups are encoded into quaternion-valued signals, followed by quaternion Fourier analysis and component-wise DE feature extraction.

2. **Quaternion Rotation Attention (QRA)**  
   Learnable quaternion rotations are applied to the value branch of attention to model subject-dependent feature misalignment.

3. **Quaternion LIF (Q-LIF)**  
   Spike firing is determined by quaternion magnitude, leading to rotation-invariant firing behavior under quaternion transformations.

---

## Why Quaternion Modeling?

Quaternion algebra provides a compact hypercomplex representation for coupling four related components. For EEG, this is useful because spatially neighboring channels often reflect coordinated neural activity rather than isolated scalar responses.

Compared with generic real-valued transformations, the Hamilton product uses structured parameter sharing. A quaternion weight has four independent components instead of a generic 4 × 4 real-valued matrix with sixteen independent parameters. This gives Q-SNet two advantages:

- **Structured coupling**: multi-channel EEG relationships are modeled explicitly.
- **Compact parameterization**: the model can maintain strong representation ability with fewer parameters.

In Q-SNet, quaternion algebra is not used as a superficial layer replacement. It is used to define the signal representation, attention transformation, rotation operation, and spiking neuron dynamics.

---

## Method Pipeline

<p align="center">
  <img src="fig3_algorithm_training.png" width="760">
</p>

The full training pipeline consists of three stages:

- **Stage 1: Quaternion-based DE feature extraction**  
  Source and target EEG trials are transformed into quaternion DE features.

- **Stage 2: Structural pre-training with QRA and Q-LIF**  
  The model learns emotion-discriminative and domain-invariant features using source labels and adversarial domain learning.

- **Stage 3: Fine-tuning with conditional alignment**  
  Target pseudo-labels are used to perform class-wise source-target alignment.

---

## Detailed Algorithms

### Quaternion-based DE Feature Extraction

<p align="center">
  <img src="fig4_algorithm_quatde.png" width="760">
</p>

This stage constructs quaternion-valued EEG signals by grouping related channels. The quaternion spectral representation is then used to compute component-wise power spectral density and differential entropy features.

### Quaternion Rotation Attention

<p align="center">
  <img src="fig5_algorithm_qra.png" width="760">
</p>

QRA replaces standard real-valued projections with quaternion projections and applies learnable quaternion rotation to the value component:

\[
V^{\circlearrowleft} = R \otimes V \otimes R^*
\]

where \(R\) is a learnable unit quaternion, \(R^*\) is its conjugate, and \(\otimes\) denotes the Hamilton product.

### Quaternion LIF Neuron

<p align="center">
  <img src="fig6_algorithm_qlif.png" width="760">
</p>

Unlike standard LIF neurons that use scalar membrane potentials, Q-LIF integrates quaternion-valued states and fires according to quaternion magnitude:

\[
S_m = \Theta(\|\tilde{U}_m\| - V_{th})
\]

Since quaternion magnitude is invariant under unit-quaternion rotations, the firing decision remains stable under rotational perturbations in quaternion feature space.

---

## Feature Distribution Visualization

<p align="center">
  <img src="figs/fig5.png" width="950">
</p>

The t-SNE visualization shows the feature evolution across different stages:

- **Raw EEG signal**: source and target samples are highly mixed.
- **Quaternion DE feature**: feature structure begins to emerge, but class separability remains limited.
- **After pre-training**: QRA and Q-LIF produce more compact feature distributions.
- **After fine-tuning**: samples from the same emotion category form clearer clusters, and source-target overlap improves.

This visualization provides intuitive evidence that Q-SNet improves both class separability and cross-subject alignment.

---

## Main Results

All experiments follow the leave-one-subject-out cross-validation protocol. One subject is used as the target domain, and the remaining subjects are used as source domains.

### Cross-subject emotion recognition performance

| Dataset | Classes | Accuracy (%) | F1 (%) | AUC (%) |
|---|---:|---:|---:|---:|
| SEED | 3 | **93.76 ± 5.08** | **93.73 ± 5.13** | **94.32 ± 3.81** |
| SEED-IV | 4 | **78.25 ± 10.89** | **76.31 ± 11.82** | **85.60 ± 7.59** |
| SEED-V | 5 | **89.98 ± 11.80** | **89.53 ± 12.57** | **93.68 ± 7.48** |

Q-SNet achieves the best performance on all three datasets. In particular, it improves over the strongest prior baseline by:

| Dataset | Previous Best | Q-SNet | Improvement |
|---|---:|---:|---:|
| SEED | 92.59 | **93.76** | +1.17 |
| SEED-IV | 76.52 | **78.25** | +1.73 |
| SEED-V | 80.21 | **89.98** | +9.77 |

---

## Ablation Study

The following ablation results show that each core component contributes to the final performance.

| Method | SEED | SEED-IV | SEED-V |
|---|---:|---:|---:|
| w/o Quaternion Self-Attention | 88.44 ± 6.42 | 66.16 ± 10.41 | 83.20 ± 12.44 |
| w/o MMD | 83.44 ± 8.69 | 69.77 ± 10.58 | 86.57 ± 10.87 |
| w/o Quaternion LIF | 88.63 ± 7.94 | 71.28 ± 12.30 | 83.25 ± 12.39 |
| Q-SNet | **93.76 ± 5.08** | **78.25 ± 10.89** | **89.98 ± 11.80** |

The performance drop after removing quaternion self-attention indicates that quaternion-valued attention is important for multi-channel representation learning. Removing Q-LIF also reduces performance, confirming the importance of magnitude-triggered spiking dynamics. Removing MMD weakens domain alignment, especially under cross-subject distribution shifts.

---

## Rotation Strategy Analysis

Q-SNet uses learnable quaternion rotations instead of fixed-axis rotations.

| Rotation Strategy | SEED | SEED-IV | SEED-V |
|---|---:|---:|---:|
| X-axis | 90.07 | 70.44 | 86.39 |
| Y-axis | 90.36 | 69.38 | 87.17 |
| Z-axis | 92.84 | 73.42 | 87.21 |
| Learnable random-axis rotation | **93.76** | **78.25** | **89.98** |

The learnable rotation consistently outperforms fixed-axis rotations, suggesting that the optimal alignment direction should be learned from data rather than predefined manually.

---

## Channel Grouping Analysis

Q-SNet constructs quaternion representations by grouping EEG channels. The grouping strategy directly affects how well inter-channel relationships are encoded.

| Grouping Strategy | SEED | SEED-IV | SEED-V |
|---|---:|---:|---:|
| Random grouping | 90.07 ± 8.89 | 70.42 ± 15.17 | 82.33 ± 10.75 |
| Original grouping | 91.49 ± 6.72 | 77.88 ± 11.35 | 88.51 ± 12.39 |
| Region-aware grouping | **93.76 ± 5.08** | **78.25 ± 10.89** | **89.98 ± 11.80** |

Region-aware grouping achieves the best performance, indicating that physiologically meaningful channel grouping improves quaternion-based EEG representation learning.

---

## Efficiency and Parameter Analysis

Q-SNet is designed to improve the trade-off between recognition performance and model compactness. The Hamilton-product-based quaternion layers reduce the number of independent parameters through structured weight sharing, while the spiking mechanism retains event-driven computation potential.

Compared with real-valued attention and Transformer variants under matched dimensions, Q-SNet provides a stronger accuracy-efficiency trade-off. This is particularly important for EEG applications where model deployment, subject adaptation, and computational cost are practical concerns.

---

## Reproducibility

This repository is organized to support reviewer-side reproducibility. It provides:

- source code for quaternion operations
- QRA and Q-LIF modules
- LOSOCV training protocol
- dataset preparation instructions
- scripts for pre-training and fine-tuning
- result logging and evaluation utilities
- figures and pseudocode used in the manuscript

Recommended repository structure:

```text
Q-SNet/
├── README.md
├── requirements.txt
├── configs/
│   ├── seed.yaml
│   ├── seed_iv.yaml
│   └── seed_v.yaml
├── data/
│   ├── SEED/
│   ├── SEED_IV/
│   └── SEED_V/
├── features/
│   ├── SEED/
│   ├── SEED_IV/
│   └── SEED_V/
├── models/
│   ├── quaternion_layers.py
│   ├── qra.py
│   ├── qlif.py
│   └── qsnet.py
├── scripts/
│   ├── extract_quaternion_de.py
│   ├── pretrain.py
│   ├── finetune.py
│   └── evaluate.py
├── utils/
│   ├── data_loader.py
│   ├── metrics.py
│   └── seed.py
└── figs/
    ├── fig1_framework.png
    ├── fig2_tsne.png
    ├── fig3_algorithm_training.png
    ├── fig4_algorithm_quatde.png
    ├── fig5_algorithm_qra.png
    └── fig6_algorithm_qlif.png
