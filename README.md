# Wood Defect Q2

Research codebase for a Q2 journal paper on wood defect detection.

## Research setup
- Main dataset: Large-scale wood surface defects
- Secondary dataset: VNWoodKnot
- Main story: small defects + efficiency + cross-dataset robustness
- Model direction: lightweight CNN backbone + 1–2 Transformer blocks + P2 small-defect branch + lightweight neck

## Planned experiment roadmap
1. Dataset audit and unified parsing
2. Small-defect definition and statistics
3. Baseline detector
4. Hybrid CNN-Transformer detector
5. Efficiency benchmark
6. Robustness benchmark
7. Cross-dataset evaluation
8. Paper asset export

## Notes
- Keep the code modular but simple.
- Prioritize reproducibility and clean experiment tracking.
- This repo is intended to be pulled to a GPU server (e.g. RTX 3090 24GB) for training and experiments.
