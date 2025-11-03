"""
p3hull
======

A research package for modeling and approximating the **P3-hull number**
in complex networks, as introduced in:

> Ponciano, V.S. & Andrade, A.  
> *A Randomized Approximation Algorithm for P3-Hull Number in Social Network Analysis* (2025)

This package includes:
- Graph generators for experimental datasets.
- Core algorithms: RAND-P3, MTS-M, and TIP-DECOMP-R.
- Tools for benchmarking and empirical evaluation.

Modules
-------
- p3hull.generators : Synthetic graph constructors.
- p3hull.algorithms : Main algorithms and evaluation utilities.

Author(s)
---------
Vitor S. Ponciano  
Axl Andrade

License
-------
Released under the MIT License.
"""

__version__ = "1.0.0"
__author__ = "Vitor S. Ponciano, Axl Andrade"
__all__ = ["generators", "algorithms"]
