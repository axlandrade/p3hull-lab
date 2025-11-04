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
            velocity: float = 1.2, seed: int = 42,
            coverage: float = 0.95) -> Tuple[Set, int]:
    """
    RAND-P3 (corrected version):
    Randomized binary-search approximation with early stopping when
    closure covers ≥ coverage * |V|.
    """

    random.seed(seed)
    nodes = list(G.nodes())
    n = len(nodes)
    if high is None:
        high = n

    weights = {v: 1.0 for v in nodes}
    best_hull, best_k, best_cov = set(), n, 0.0

    while low <= high:
        k = (low + high) // 2
        found, S = exists_hull(G, k, weights, samples, seed + k)
        H = p3_closure(G, S)
        cov = len(H) / n

        # early stop if closure covers enough vertices
        if cov >= coverage:
            best_hull, best_k, best_cov = S, k, cov
            high = k - 1  # try smaller k
            if weighted:
                for v in S:
                    weights[v] *= velocity
        else:
            low = k + 1

    # final refinement: shrink best_hull if closure already near total
    if best_cov >= coverage:
        refined = sorted(best_hull)
        for v in list(refined):
            test = set(refined)
            test.remove(v)
            if len(p3_closure(G, test)) >= coverage * n:
                refined.remove(v)
        best_hull = set(refined)
        best_k = len(best_hull)

    return best_hull, best_k


# ---------------------------------------------------------------------
# 5. TIP-DECOMP-R algorithm (randomized)
# ---------------------------------------------------------------------

def tip_decomp_random(G: nx.Graph, threshold: int = 2, seed: int = 42) -> set:
    """
    TIP-DECOMP-R as a TARGET-SET heuristic (what Table 7 reports).
    Undirected graph, threshold=2 (P3-convexity).

    Rules:
      - If deg(v) < t(v), v must be in the seed set S.
      - Otherwise remove a vertex with minimal (deg(v) - t(v)) (random tie).
    """
    import random
    random.seed(seed)

    H = G.copy()
    # threshold t(v)=2 for all; we only need degrees in H
    t = {v: threshold for v in H.nodes()}
    S = set()                 # <- THIS is what Table 7 reports
    remaining = set(H.nodes())

    while remaining:
        # recompute degrees on the current graph
        deg = {v: H.degree(v) for v in remaining}

        # Case A: vertices that cannot be influenced by remaining (deg < t)
        forced = [v for v in remaining if deg[v] < t[v]]
        if forced:
            v = min(forced, key=lambda x: deg[x])  # deterministic pick among forced
            S.add(v)
            # seeding v reduces neighbors' thresholds by 1 (they see one active neighbor)
            for u in list(H.neighbors(v)):
                if u in remaining:
                    t[u] = max(0, t[u] - 1)
            H.remove_node(v)
            remaining.remove(v)
            continue

        # Case B: no forced vertices -> peel one with minimal (deg - t)
        scores = {v: deg[v] - t[v] for v in remaining}
        min_score = min(scores.values())
        candidates = [v for v in remaining if scores[v] == min_score]
        v = random.choice(candidates)              # <- random tie-break (TIP-DECOMP-R)

        # peeling v (it will be activated later by others)
        for u in list(H.neighbors(v)):
            if u in remaining:
                # edge removal reduces neighbor degree only; thresholds stay
                pass
        H.remove_node(v)
        remaining.remove(v)

    return S

def tip_decomp_deterministic(G, threshold=2):
    # mesma função, mas v = min(candidates) no caso B
    import random
    H = G.copy()
    t = {v: threshold for v in H.nodes()}
    S = set()
    remaining = set(H.nodes())
    while remaining:
        deg = {v: H.degree(v) for v in remaining}
        forced = [v for v in remaining if deg[v] < t[v]]
        if forced:
            v = min(forced, key=lambda x: deg[x])
            S.add(v)
            for u in list(H.neighbors(v)):
                if u in remaining:
                    t[u] = max(0, t[u] - 1)
            H.remove_node(v); remaining.remove(v)
            continue
        scores = {v: deg[v] - t[v] for v in remaining}
        min_score = min(scores.values())
        candidates = [v for v in remaining if scores[v] == min_score]
        v = min(candidates)  # determinístico
        H.remove_node(v); remaining.remove(v)
    return S

# ---------------------------------------------------------------------
# 6. MTS-M algorithm (modified Minimum Target Set)
# ---------------------------------------------------------------------

def mts_m(G: nx.Graph, threshold: int = 2) -> set:
    """
    Modified Minimal Target Set (MTS-M)
    Variante que prioriza vértices com maior grau entre os forçados.
    """
    H = G.copy()
    S = set()
    t = {v: threshold for v in H.nodes()}
    remaining = set(H.nodes())

    while remaining:
        deg = {v: H.degree(v) for v in remaining}
        forced = [v for v in remaining if deg[v] < t[v]]
        if forced:
            v = max(forced, key=lambda x: deg[x])  # diferença: maior grau
            S.add(v)
            for u in list(H.neighbors(v)):
                if u in remaining:
                    t[u] = max(0, t[u] - 1)
            H.remove_node(v)
            remaining.remove(v)
            continue

        scores = {v: deg[v] - t[v] for v in remaining}
        min_score = min(scores.values())
        candidates = [v for v in remaining if scores[v] == min_score]
        v = max(candidates)  # diferença: escolhe o de maior ID
        H.remove_node(v)
        remaining.remove(v)

    return S


def mts(G: nx.Graph, threshold: int = 2) -> set:
    """
    Minimal Target Set heuristic (Ponciano & Andrade, 2025).
    Identifica o menor conjunto de vértices capaz de ativar toda a rede.
    """
    H = G.copy()
    S = set()
    t = {v: threshold for v in H.nodes()}
    remaining = set(H.nodes())

    while remaining:
        # Recalcula graus
        deg = {v: H.degree(v) for v in remaining}

        # Caso 1: vértices que não podem ser ativados por vizinhos
        forced = [v for v in remaining if deg[v] < t[v]]
        if forced:
            v = min(forced, key=lambda x: deg[x])  # determinístico
            S.add(v)
            for u in list(H.neighbors(v)):
                if u in remaining:
                    t[u] = max(0, t[u] - 1)
            H.remove_node(v)
            remaining.remove(v)
            continue

        # Caso 2: nenhum vértice forçado — remove o de menor (grau - limiar)
        scores = {v: deg[v] - t[v] for v in remaining}
        min_score = min(scores.values())
        candidates = [v for v in remaining if scores[v] == min_score]
        v = min(candidates)
        H.remove_node(v)
        remaining.remove(v)

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
