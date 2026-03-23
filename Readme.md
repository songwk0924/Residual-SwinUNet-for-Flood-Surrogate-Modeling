# Residual Swin-UNet for Cross-Regional Flood Surrogate Modeling

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.watres.2026.125799-blue)](https://doi.org/10.1016/j.watres.2026.125799)

Official implementation of the core model architecture and transfer learning utilities for the paper published in **Water Research**:
> **"Enhancing cross-regional transferability of super-resolution-based flood surrogate models for data-scarce catchments."**

**Authors:** Wenke Song, Mingfu Guan  
**Affiliation:** Department of Civil Engineering, The University of Hong Kong  
**Contact:** [songwk@connect.hku.hk](mailto:songwk@connect.hku.hk)

---

## 📝 Abstract

Deep learning-based flood surrogate models have shown promise in accelerating spatiotemporal flood simulations, yet their cross-regional transferability remains a significant challenge, limiting widespread application in data-scarce catchments. This study proposes a transfer learning framework to address the transferability of flood surrogate models across catchments with inconsistent base resolutions and diverse terrain features. Within this framework, flood surrogate modeling is considered as an image super-resolution task. Specifically, a Residual SwinUNet model is developed to fuse multi channel coarse grid flood maps with fine‑grid topographic features for high‑resolution flood map reconstruction. The framework is applied to the upper Shenzhen River catchment in China and Richmond River catchment in Australia, using various transfer learning strategies. Results demonstrate that under data-abundant conditions, the proposed model accurately reconstructs high-resolution flood maps for both uniform and spatiotemporally distributed rainfall events. In data-scarce scenarios, full fine-tuning strategy recovers over 90% of baseline accuracy using less than 3% of the pre-training events, while Low-Rank Adaptation matches or even surpasses scratch training performance with only 10–35% of trainable parameters. Furthermore, cross-scale experiments reveal that the framework is effective across a broad range of scale factors. Pretraining on smaller source scale factors enhances transferability, while larger target scale factors increase the relative advantage of transfer learning. In addition, balancing fine-tuning events across duration and cumulative depth improves transfer robustness. This study establishes a new paradigm for enhancing cross-regional adaptation of flood surrogate models for data-scarce catchments.

**Keywords:** Cross-regional transferability, Rapid flood simulation, Image super-resolution, Transfer learning, Residual SwinUNet, LoRA.

---

## 📂 Repository Contents

This repository is designed to be lightweight and focuses on providing the core model structures proposed in the paper.

*   `Residual_SwinUNet.py`: The architecture of the **Residual Swin-UNet**, a hybrid CNN-Swin Transformer-based U-Shaped network.
*   `lora_utils.py`: A plug-and-play implementation of **Low-Rank Adaptation (LoRA)** designed to work seamlessly with the Residual Swin-UNet for efficient fine-tuning.

## 📌 Citation

If you find this code or our paper useful for your research, please cite our work:

**APA Format:**
> Song, W., & Guan, M. (2026). Enhancing cross-regional transferability of super-resolution-based flood surrogate models for data-scarce catchments. *Water Research*, 125799. https://doi.org/10.1016/j.watres.2026.125799
