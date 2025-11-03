"""
experiments.evaluation
======================

Automated benchmarking script for the algorithms in `p3hull.algorithms`
on multiple synthetic graph families defined in `p3hull.generators`.

It measures execution time and hull size for:
- RAND-P3
- MTS-M
- TIP-DECOMP-R

Outputs results as CSV for reproducibility and figure generation.
"""

import time
import pandas as pd
import networkx as nx
from p3hull.generators import multiple_graph, inflated_cycle, bipartite_odd, grid_2d
from p3hull.algorithms import evaluate_all

def run_benchmark(output_csv: str = "benchmark_results.csv"):
    results = []

    # Define graph families and parameters
    test_graphs = [
        ("M(50,5)", multiple_graph(50, 5)),
        ("InflatedCycle(20,3)", inflated_cycle(20, 3)),
        ("Bipartite(2t+1)", bipartite_odd(10)),
        ("Grid2D(10,10)", grid_2d(10, 10)),
    ]

    for name, G in test_graphs:
        print(f"\n=== Evaluating {name} ===")
        start_total = time.time()
        res = evaluate_all(G)
        total_time = time.time() - start_total

        for alg_name, (S, k) in res.items():
            results.append({
                "graph": name,
                "algorithm": alg_name,
                "hull_size": k,
                "n": G.number_of_nodes(),
                "m": G.number_of_edges(),
                "avg_degree": 2*G.number_of_edges()/G.number_of_nodes(),
                "runtime_sec": total_time
            })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Results saved to {output_csv}")
    return df


if __name__ == "__main__":
    run_benchmark()
