# V-STAR: Value-guided Sampling and Tree-structured Advantage Reinforcement

Official implementation for the KDD 2026 paper:

> **Spend Search Where It Pays: Value-Guided Structured Sampling and Optimization for Generative Recommendation**


---

## Overview

V-STAR addresses the **probability-reward mismatch** in generative recommendation by coupling two synergistic components:

- **Value-Guided Efficient Decoding (VED)**: A budgeted tree-search algorithm that allocates decoding compute to high-value, high-uncertainty prefixes, improving reachability of long-tail items without exhaustive search.
- **Sibling-GRPO**: A tree-structured policy optimization objective that computes sibling-relative advantages at each branching depth, restoring informative learning signals under prefix coupling.

Together, they form a self-evolving loop: VED improves candidate quality, and Sibling-GRPO converts these gains into more stable policy updates.

---

## Quick Start

### 1. Environment Setup

```bash
conda create -n vstar python=3.11 -y
conda activate vstar
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
bash convert_dataset.sh
```

Or follow the full pipeline in `data/` and `rq/` for raw data processing and SID construction.

### 3. Supervised Fine-Tuning (SFT)

```bash
bash sft.sh
```

### 4. V-STAR Training (VED + Sibling-GRPO)

```bash
bash run_vstar.sh
```

### 5. Evaluation

```bash
bash evaluate.sh
```

---

## Training Pipeline

```
Data Preprocessing ─→ SID Construction (RQ-VAE) ─→ SFT ─→ V-STAR (VED + Sibling-GRPO) ─→ Evaluation
     data/                    rq/                  sft.sh      run_vstar.sh              evaluate.sh
```

---

## Requirements

- GPUs: 8 × A100 (80 GB) recommended
- Python: 3.11
- Key packages: transformers, trl, deepspeed, accelerate, torch

---

## Citation

```bibtex
@inproceedings{jiang2026vstar,
  title={Spend Search Where It Pays: Value-Guided Structured Sampling and Optimization for Generative Recommendation},
  author={Jiang, Jie and Huang, Yangru and Wang, Zeyu and Wang, Changping and Xiong, Yuling and Zhang, Jun and Yu, Huan},
  booktitle={Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year={2026}
}
```

---

## Acknowledgements

This codebase builds upon [MiniOneRec](https://github.com/AkaliKong/MiniOneRec). We thank the authors for their open-source framework.

---

## License

All Rights Reserved.
