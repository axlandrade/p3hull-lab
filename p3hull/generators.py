"""
p3hull.generators
=================

Graph generators used for testing and evaluating the P3-hull algorithms
presented in the paper "A Randomized Approximation Algorithm for P3-Hull Number
in Social Network Analysis" (Ponciano et al., 2025).

This module consolidates all previously separate scripts and notebooks into
reusable NetworkX-based functions for synthetic graph generation.

Each function returns a NetworkX Graph ready for use in the RAND-P3, MTS-M,
and TIP-DECOMP-R algorithms.

Author: Axl Andrade & Vitor S. Ponciano
"""

import networkx as nx
import itertools


# ---------------------------------------------------------------------
# 1. Multiple Graphs M(n, q)
# ---------------------------------------------------------------------

def multiple_graph(n: int, q: int) -> nx.Graph:
    """
    Generate a multiple graph M(n, q) as defined in the paper.

    Parameters
    ----------
    n : int
        Total number of nodes in the u-layer.
    q : int
        Number of cliques (and also determines the clique size).

    Returns
    -------
    G : networkx.Graph
        The generated M(n, q) graph.

    Notes
    -----
    - Vertices are named as 'u1', 'u2', ..., 'un' and 'v1', 'v2', ..., 'vn'.
    - Edges:
        * Each pair (u_i, v_i)
        * A cycle connecting all u_i
        * Cliques formed among v_j vertices spaced by q
    """
    G = nx.Graph()

    # Layer u (cycle + u-v connections)
    for i in range(1, n + 1):
        G.add_edge(f"u{i}", f"v{i}")
        if i < n:
            G.add_edge(f"u{i}", f"u{i+1}")
    G.add_edge(f"u{n}", "u1")  # close the cycle

    # Create q cliques among v-layer vertices
    k = n // q
    for start in range(q):
        clique = [f"v{start + 1 + j*q}" for j in range(k) if start + 1 + j*q <= n]
        for u, v in itertools.combinations(clique, 2):
            G.add_edge(u, v)

    return G


# ---------------------------------------------------------------------
# 2. Inflated Cycle
# ---------------------------------------------------------------------

def inflated_cycle(k: int, r: int) -> nx.Graph:
    """
    Generate an inflated cycle graph C_k[r], where each vertex of a
    k-cycle is replaced by a clique of size r.

    Parameters
    ----------
    k : int
        Number of cycle vertices (the base cycle length).
    r : int
        Size of each inflated clique.

    Returns
    -------
    G : networkx.Graph
        The inflated cycle graph.
    """
    G = nx.Graph()
    for i in range(k):
        clique = [f"c{i}_{j}" for j in range(r)]
        for u, v in itertools.combinations(clique, 2):
            G.add_edge(u, v)
        # Connect each clique to the next (cyclically)
        next_clique = [f"c{(i+1)%k}_{j}" for j in range(r)]
        for u in clique:
            G.add_edge(u, next_clique[0])
    return G


# ---------------------------------------------------------------------
# 3. Bipartite Graph B(2t + 1)
# ---------------------------------------------------------------------

def bipartite_odd(t: int) -> nx.Graph:
    """
    Generate a simple bipartite graph with 2t + 1 vertices.

    Parameters
    ----------
    t : int
        Controls the size of the bipartite parts (graph will have 2t + 1 nodes).

    Returns
    -------
    G : networkx.Graph
        Bipartite graph with uneven partition.
    """
    G = nx.Graph()
    left = [f"L{i}" for i in range(1, t + 1)]
    right = [f"R{i}" for i in range(1, t + 2)]
    G.add_nodes_from(left, bipartite=0)
    G.add_nodes_from(right, bipartite=1)

    # Fully connect L to R (almost complete bipartite)
    for u in left:
        for v in right:
            G.add_edge(u, v)
    return G


# ---------------------------------------------------------------------
# 4. Partial and 3D Grids
# ---------------------------------------------------------------------

def grid_2d(n: int, m: int) -> nx.Graph:
    """Return an n-by-m 2D grid graph."""
    return nx.grid_2d_graph(n, m)


def grid_3d(n: int, m: int, p: int) -> nx.Graph:
    """Return an n-by-m-by-p 3D grid graph."""
    return nx.grid_graph([range(n), range(m), range(p)])


def partial_grid(n: int, m: int, removal_prob: float = 0.2, seed: int = 42) -> nx.Graph:
    """
    Generate a partial grid by randomly removing edges from a 2D grid.

    Parameters
    ----------
    n, m : int
        Grid dimensions.
    removal_prob : float
        Probability of removing each edge.
    seed : int
        Random seed for reproducibility.
    """
    import random
    random.seed(seed)
    G = grid_2d(n, m)
    for e in list(G.edges()):
        if random.random() < removal_prob:
            G.remove_edge(*e)
    return G


# ---------------------------------------------------------------------
# 5. Trees
# ---------------------------------------------------------------------

def random_tree(n: int, seed: int = 42) -> nx.Graph:
    """
    Generate a random tree using NetworkX's random_tree function.

    Parameters
    ----------
    n : int
        Number of vertices.
    seed : int
        Random seed.
    """
    return nx.random_tree(n, seed=seed)
