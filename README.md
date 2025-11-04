# p3hull-lab

A research toolkit for analyzing and approximating the **P3-hull number** in social networks, implementing the algorithms introduced in:

> **Ponciano, V.S. & Hora, Y. & Dantas, S. & Andrade, A.**  
> *A Randomized Approximation Algorithm for P3-Hull Number in Social Network Analysis* (2025)

---

## Overview

`p3hull-lab` provides an experimental and reproducible environment for evaluating algorithms that estimate the **P3-hull number** — the smallest set of vertices whose P3-closure covers all nodes of a graph.

The repository includes:

- **Graph Generators** – synthetic families like Multiple Graphs `M(n, q)`, Inflated Cycles, Grids, and Bipartite networks.  
- **Algorithms** – `RAND-P3`, `MTS-M`, and `TIP-DECOMP-R` implementations.  
- **Benchmark Suite** – automated evaluation, statistical analysis, and visualization tools.  

---

## Repository Structure

```
p3hull-lab/
│
├── p3hull/
│   ├── __init__.py
│   ├── generators.py        # Graph generators
│   ├── algorithms.py        # Main algorithms (RAND-P3, MTS-M, TIP-DECOMP-R)
│   └── utils.py             # (optional) helper functions
│
├── experiments/
│   ├── evaluation.py        # Benchmark execution script
│   ├── benchmark.ipynb      # Result analysis and visualization
│   └── data/                # GraphML / CSV outputs
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

### Option 1 — Local environment

```bash
git clone https://github.com/axlandrade/p3hull-lab.git
cd p3hull-lab
pip install -r requirements.txt
```

### Option 2 — Editable installation for development

```bash
pip install -e .
```

---

## Usage

### Example: Running the benchmark suite

```bash
python experiments/evaluation.py
```

This script:

- Generates multiple synthetic graph families;
- Runs the algorithms (`RAND-P3`, `MTS-M`, `TIP-DECOMP-R`);
- Saves results to `benchmark_results.csv`.

### Example: Visualizing results

Open the notebook:

```bash
jupyter notebook experiments/benchmark.ipynb
```

You’ll see bar plots comparing **hull size** and **runtime** across algorithms and graph families.

---

## Algorithms Implemented

| Algorithm        | Description                                               | Complexity            | Determinism   |
| ---------------- | --------------------------------------------------------- | --------------------- | ------------- |
| **RAND-P3**      | Randomized binary search with adaptive weighted sampling. | ~O(s·n) per iteration | Probabilistic |
| **MTS-M**        | Modified Minimum Target Set (threshold t=2).              | O(n·d)                | Deterministic |
| **TIP-DECOMP-R** | Randomized decomposition heuristic.                       | O(n log n)            | Probabilistic |

---

## Supported Graph Families

| Generator              | Function                           | Description                      |
| ---------------------- | ---------------------------------- | -------------------------------- |
| `multiple_graph(n, q)` | Multiple Graph M(n, q)             | Used in original experiments     |
| `inflated_cycle(k, r)` | Inflated cycle Cₖ[r]               | Each vertex replaced by a clique |
| `bipartite_odd(t)`     | Bipartite graph with 2t+1 vertices | Uneven partition                 |
| `grid_2d(n, m)`        | 2D lattice                         | Regular grid graph               |
| `grid_3d(n, m, p)`     | 3D lattice                         | Extension to volumetric networks |
| `random_tree(n)`       | Random spanning tree               | Sparse acyclic instance          |

---

## Citation

If you use this repository or its algorithms in academic work, please cite:

```
@article{ponciano,
  title={A Randomized Approximation Algorithm for P3-Hull Number in Social Network Analysis},
  author={Ponciano, Vitor S. and Dantas, Simone and Hora, Ygor},
  year={2025},
  note={Manuscript in preparation}
}
```

---

## Authors

- **Vitor S. Ponciano**
- **Axl Andrade**
- **Ygor Hora**
- **Simone Dantas** 

---

## License

This project is released under the **MIT License**.
You are free to use, modify, and distribute it for research or educational purposes.