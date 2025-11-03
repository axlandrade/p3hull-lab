"""
p3hull.algorithms
=================

Implementation of the main algorithms for the P3-hull number problem
used in the paper:

"A Randomized Approximation Algorithm for P3-Hull Number
 in Social Network Analysis" (Ponciano et al., 2025).

Algorithms implemented:
-----------------------
- RAND-P3: Randomized approximation algorithm using binary search and
  weighted random sampling.
- MTS-M: Modified Minimum Target Set heuristic.
- TIP-DECOMP-R: Randomized version of TIP-DECOMP.

All algorithms assume the graph is undirected and unweighted.
"""

from __future__ import annotations
import networkx as nx
import random
import itertools
from typing import Set, Dict, Tuple, Optional, List


# ---------------------------------------------------------------------
# 1. Core P3-closure function
# ---------------------------------------------------------------------

def p3_closure(G: nx.Graph, S: Set) -> Set:
    """
    Compute the P3-closure of a set S in graph G.

    A vertex becomes activated if it has at least two active neighbors.
    The process repeats until no new vertices can be activated.

    Parameters
    ----------
    G : networkx.Graph
        Input undirected graph.
    S : set
        Initial seed set.

    Returns
    -------
    H : set
        P3-closure of S (i.e., all activated vertices).
    """
    activated = set(S)
    while True:
        new_active = set()
        for v in G.nodes():
            if v in activated:
                continue
            active_neighbors = len([u for u in G.neighbors(v) if u in activated])
            if active_neighbors >= 2:
                new_active.add(v)
        if not new_active:
            break
        activated |= new_active
    return activated


# ---------------------------------------------------------------------
# 2. Weighted random sampling (Efraimidis–Spirakis)
# ---------------------------------------------------------------------

def weighted_random_sample(weights: Dict, k: int) -> List:
    """
    Sample k distinct elements based on given weights using
    the Efraimidis–Spirakis weighted random sampling scheme.

    Parameters
    ----------
    weights : dict
        Mapping vertex -> weight (non-negative).
    k : int
        Sample size.

    Returns
    -------
    sample : list
        List of selected vertices.
    """
    keys = []
    for v, w in weights.items():
        if w <= 0:
            continue
        r = random.random()
        keys.append((r ** (1.0 / w), v))
    keys.sort(reverse=True)
    return [v for (_, v) in keys[:k]]


# ---------------------------------------------------------------------
# 3. Check for existence of a hull set of size k
# ---------------------------------------------------------------------

def exists_hull(G: nx.Graph, k: int, weights: Optional[Dict] = None,
                samples: int = 200, seed: int = 42) -> Tuple[bool, Set]:
    """
    Check (probabilistically) whether a P3-hull set of size k exists.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    k : int
        Candidate hull size.
    weights : dict, optional
        Probability weights for vertex sampling.
    samples : int
        Number of random samples to test.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    exists : bool
        True if a hull set of size k was found.
    best_set : set
        Best (largest closure) set found.
    """
    random.seed(seed)
    nodes = list(G.nodes())
    if weights is None:
        weights = {v: 1.0 for v in nodes}

    best_set = set()
    best_closure_size = 0

    for _ in range(samples):
        S = set(weighted_random_sample(weights, k))
        H = p3_closure(G, S)
        if len(H) > best_closure_size:
            best_set = S
            best_closure_size = len(H)
        if len(H) == len(G):
            return True, S  # perfect hull found

    return best_closure_size == len(G), best_set


# ---------------------------------------------------------------------
# 4. RAND-P3 algorithm
# ---------------------------------------------------------------------

def rand_p3(G: nx.Graph, low: int = 1, high: Optional[int] = None,
            samples: int = 200, weighted: bool = True,
            velocity: float = 1.2, seed: int = 42) -> Tuple[Set, int]:
    """
    RAND-P3: Randomized binary-search approximation for the P3-hull number.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    low : int
        Minimum hull size candidate.
    high : int, optional
        Maximum hull size candidate (defaults to |V|).
    samples : int
        Number of random samples per iteration.
    weighted : bool
        Whether to use adaptive weighting.
    velocity : float
        Multiplicative factor for successful vertices' weights.
    seed : int
        Random seed.

    Returns
    -------
    best_hull : set
        Best hull set found.
    hull_size : int
        Approximated P3-hull number.
    """
    random.seed(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    if high is None:
        high = n

    weights = {v: 1.0 for v in nodes}
    best_hull, best_k = set(), n

    while low <= high:
        k = (low + high) // 2
        found, S = exists_hull(G, k, weights, samples, seed + k)
        if found:
            best_hull, best_k = S, k
            high = k - 1  # try smaller
            if weighted:
                for v in S:
                    weights[v] *= velocity
        else:
            low = k + 1

    return best_hull, best_k


# ---------------------------------------------------------------------
# 5. TIP-DECOMP-R algorithm (randomized)
# ---------------------------------------------------------------------

def tip_decomp_random(G: nx.Graph, seed: int = 42) -> Set:
    """
    Randomized version of the TIP-DECOMP algorithm.

    Removes vertices iteratively, randomly selecting among those with
    minimal distance values, until a stable core remains.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    seed : int
        Random seed.

    Returns
    -------
    core : set
        Remaining vertices after decomposition.
    """
    random.seed(seed)
    G = G.copy()
    dist = {v: G.degree(v) - 2 for v in G.nodes()}
    remaining = set(G.nodes())

    while remaining:
        min_dist = min(dist[v] for v in remaining)
        candidates = [v for v in remaining if dist[v] == min_dist]
        v = random.choice(candidates)
        remaining.remove(v)
        for u in list(G.neighbors(v)):
            if u in remaining:
                dist[u] = max(0, dist[u] - 1)
        G.remove_node(v)

    return set(G.nodes())


# ---------------------------------------------------------------------
# 6. MTS-M algorithm (modified Minimum Target Set)
# ---------------------------------------------------------------------

def mts_m(G: nx.Graph) -> Set:
    """
    Simplified and modified version of the MTS algorithm
    (Cordasco et al., 2016), adapted for undirected graphs
    and threshold t(v) = 2 for all vertices.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.

    Returns
    -------
    S : set
        Target set (approximate hull set).
    """
    G = G.copy()
    t = {v: 2 for v in G.nodes()}     # threshold
    delta = {v: G.degree(v) for v in G.nodes()}
    U, S, L = set(G.nodes()), set(), set()

    while U:
        # Case 1: vertex already influenced
        case1 = [v for v in U if t[v] <= 0]
        if case1:
            v = case1[0]
            for u in G.neighbors(v):
                if u in U:
                    t[u] = max(t[u] - 1, 0)
            U.remove(v)
            continue

        # Case 2: vertex cannot be influenced by remaining
        case2 = [v for v in U if delta[v] < t[v]]
        if case2:
            v = min(case2, key=lambda x: delta[x])
            S.add(v)
            for u in G.neighbors(v):
                if u in U:
                    t[u] = max(t[u] - 1, 0)
                    delta[u] = max(delta[u] - 1, 0)
            U.remove(v)
            continue

        # Case 3: pick vertex with minimal score (heuristic)
        v = min(U, key=lambda x: t[x] * (delta[x] + 1))
        for u in G.neighbors(v):
            if u in U:
                delta[u] = max(delta[u] - 1, 0)
        L.add(v)
        U.remove(v)

    return S


# ---------------------------------------------------------------------
# 7. Utility: run all algorithms and compare
# ---------------------------------------------------------------------

def evaluate_all(G: nx.Graph, seed: int = 42) -> Dict[str, Tuple[Set, int]]:
    """
    Run RAND-P3, TIP-DECOMP-R, and MTS-M on the same graph.

    Parameters
    ----------
    G : networkx.Graph
        Input graph.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Mapping from algorithm name to (set, size) tuple.
    """
    randp3_set, randp3_k = rand_p3(G, seed=seed)
    tip_set = tip_decomp_random(G, seed)
    mts_set = mts_m(G)
    return {
        "RAND-P3": (randp3_set, randp3_k),
        "TIP-DECOMP-R": (tip_set, len(tip_set)),
        "MTS-M": (mts_set, len(mts_set))
    }
