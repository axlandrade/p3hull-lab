"""
Enhanced benchmark execution using the existing RAND-P3 implementation
without modifying the original algorithm code.
"""

import time
import pandas as pd
from p3hull.algorithms import rand_p3, mts_m, tip_decomp_random, p3_closure
from p3hull.generators import multiple_graph, inflated_cycle, bipartite_odd, grid_2d
import networkx as nx

# --- 1. Pré-processamento híbrido: usar MTS-M como base de pesos
def prepare_weights(G):
    seed = mts_m(G)
    weights = {v: 2.5 if v in seed else 1.0 for v in G.nodes()}
    return weights


# --- 2. Pós-processamento: remoção local de vértices redundantes
def local_refinement(G, S):
    S = set(S)
    for v in list(S):
        if len(p3_closure(G, S - {v})) == len(G):
            S.remove(v)
    return S


# --- 3. Execução otimizada dos algoritmos
def run_enhanced_benchmark():
    results = []
    graphs = [
        ("M(50,5)", multiple_graph(50, 5)),
        ("InflatedCycle(20,3)", inflated_cycle(20, 3)),
        ("Bipartite(2t+1)", bipartite_odd(10)),
        ("Grid2D(10,10)", grid_2d(10, 10)),
    ]

    for name, G in graphs:
        print(f"\n=== Running enhanced RAND-P3 on {name} ===")
        weights = prepare_weights(G)

        # RAND-P3 com parâmetros mais exploratórios
        start = time.time()
        S, k = rand_p3(G, samples=1000, velocity=2.0, weighted=True, seed=42)
        S = local_refinement(G, S)
        runtime = time.time() - start

        results.append({
            "graph": name,
            "algorithm": "RAND-P3 (enhanced)",
            "hull_size": len(S),
            "n": len(G),
            "m": G.number_of_edges(),
            "runtime_sec": runtime
        })

    df = pd.DataFrame(results)
    df.to_csv("benchmark_enhanced_results.csv", index=False)
    print("\n✅ Saved as benchmark_enhanced_results.csv")
    return df


if __name__ == "__main__":
    run_enhanced_benchmark()
