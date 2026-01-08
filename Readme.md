# Residual Swin-UNet for Cross-Regional Flood Surrogate Modeling

Official implementation of the core model architecture and transfer learning utilities for the paper:
> **"Enhancing Cross-Regional Transferability of Deep Learning-Based Flood Surrogate Models to Data-Scarce Catchments"**

**Authors:** Wenke Song, Mingfu Guan  
**Affiliation:** Department of Civil Engineering, The University of Hong Kong  
**Contact:** [songwk@connect.hku.hk](mailto:songwk@connect.hku.hk)

---

## 📝 Abstract

Deep learning-based flood surrogate models have shown promise in accelerating spatiotemporal flood simulations, yet their cross-regional transferability remains a significant challenge, limiting widespread application in data-scarce catchments. This study proposes a robust transfer learning framework to address the transferability of flood surrogate models across catchments with inconsistent base resolutions and diverse terrain features.

Within this framework, a **Residual Swin-UNet** model is developed to fuse coarse‑grid hydrodynamic temporal context with fine‑grid topographic features for high‑resolution flood map reconstruction. Results demonstrate that in data-scarce scenarios, **Low-Rank Adaptation (LoRA)** matches or even surpasses scratch training performance with only 10–35% of trainable parameters. This study establishes a new paradigm for enhancing cross-regional adaptation of flood surrogate models for data-scarce catchments.

**Keywords:** Cross-regional transferability, rapid flood simulation, transfer learning, Residual SwinUNet, LoRA.

---

## 📂 Repository Contents

This repository is designed to be lightweight and focuses on providing the core model structures proposed in the paper.

*   `Residual_SwinUNet.py`: The architecture of the **Residual Swin-UNet**, a hybrid CNN-Swin Transformer-based U-Shaped network.
*   `lora_utils.py`: A plug-and-play implementation of **Low-Rank Adaptation (LoRA)** designed to work seamlessly with the Residual Swin-UNet for efficient fine-tuning.
