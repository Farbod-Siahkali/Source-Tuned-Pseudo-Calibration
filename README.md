# Coverage Guarantees for Pseudo-Calibrated Conformal Prediction

Code and numerical results for the paper
[Coverage Guarantees for Pseudo-Calibrated Conformal Prediction under Distribution Shift](https://arxiv.org/abs/2602.14913).

The `notebooks/` directory contains the output-cleared experiment workflows. Run each notebook from top to bottom; the datasets, including CIFAR-10-C, are downloaded automatically.

| Notebook | Results |
| --- | --- |
| `mnist.ipynb`, `cifar10.ipynb`, `cifar100.ipynb` | Table I (five runs) and Figs. 1-2 |
| `mnist_supplement.ipynb`, `cifar10_supplement.ipynb` | Sweeps over the nominal level and the MNIST-C/CIFAR-10-C benchmarks (extended version) |
| `mnist_covdiff.ipynb`, `cifar10_covdiff.ipynb` | Coverage difference versus classifier loss and shift level (extended version) |

The ECP and WQLCP comparators are implemented in `notebooks/baselines.py`. The MNIST-C severities are generated with the `shot_noise` function of the official MNIST-C code.

The `results/` directory contains the figures used in the paper and the numerical values from Table I.

## Setup

```bash
python -m pip install -r requirements.txt
jupyter lab
```

A CUDA-capable GPU is recommended, especially for CIFAR-10 and CIFAR-100. The figures were produced with matplotlib 3.4.3, which is pinned because the layout of the 3D figures depends on the version.

## Notes

- GPU training of the CIFAR classifiers is not bit-for-bit reproducible, so CIFAR results can differ slightly between runs and hardware. The MNIST values in Table I are reproduced exactly.
- The conformal quantile helper returns the order statistic one rank above the $\lceil (n+1)(1-\alpha) \rceil$-th, which is slightly more conservative than Eq. (2) (by at most $1/n$ in coverage).

## Citation

```bibtex
@article{siahkali2026coverage,
  title   = {Coverage Guarantees for Pseudo-Calibrated Conformal Prediction under Distribution Shift},
  author  = {Siahkali, Farbod and Verma, Ashwin and Gupta, Vijay},
  journal = {arXiv preprint arXiv:2602.14913},
  year    = {2026}
}
```
