# Coverage Guarantees for Pseudo-Calibrated Conformal Prediction

Code and numerical results for the paper
[Coverage Guarantees for Pseudo-Calibrated Conformal Prediction under Distribution Shift](https://arxiv.org/abs/2602.14913).

The `notebooks/` directory contains the output-cleared, dataset-specific experiment workflows for MNIST, CIFAR-10, and CIFAR-100. Run each notebook from top to bottom; the standard torchvision datasets are downloaded automatically. The ECP and WQLCP comparators are implemented in `notebooks/baselines.py`.

The `results/` directory contains the final five-run figures used in the paper, the additional revision experiments, and the numerical values from the main results table.

## Setup

```bash
python -m pip install -r requirements.txt
jupyter lab
```

A CUDA-capable GPU is recommended, especially for CIFAR-10 and CIFAR-100.

## Citation

```bibtex
@article{siahkali2026coverage,
  title   = {Coverage Guarantees for Pseudo-Calibrated Conformal Prediction under Distribution Shift},
  author  = {Siahkali, Farbod and Verma, Ashwin and Gupta, Vijay},
  journal = {arXiv preprint arXiv:2602.14913},
  year    = {2026}
}
```
