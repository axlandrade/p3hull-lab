"""
experiments/full_eval_tgf.py
---------------------------------
Parallel benchmark reproducing Table 7 of Ponciano & Andrade (2025)
"""

from pathlib import Path
from multiprocessing import Pool, cpu_count
import pandas as pd
from tqdm import tqdm
import time
import networkx as nx

from p3hull.algorithms import rand_p3, mts_m, tip_decomp_random, mts
from p3hull.algorithms import tip_decomp_deterministic  # certifique-se de ter essa função

# ---------------------------------------------
# 1. Leitura dos grafos .tgf (não-direcionado)
# ---------------------------------------------
def load_tgf(path):
    """Load a .tgf as an undirected, unweighted graph."""
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = s.split()
            if len(parts) >= 2:
                u, v = parts[:2]
                G.add_edge(u, v)
    return G


# ---------------------------------------------
# 2. Avaliação de um único grafo
# ---------------------------------------------
def evaluate_algorithms(G):
    """Executa os algoritmos sobre um grafo."""
    results = {}

    # ---- MTS (original) ----
    start = time.time()
    S_mts = mts(G)
    results["MTS"] = len(S_mts)
    results["runtime_MTS"] = time.time() - start

    # ---- MTS-M (modificado) ----
    start = time.time()
    S_mtsm = mts_m(G)
    results["MTS-M"] = len(S_mtsm)
    results["runtime_MTSM"] = time.time() - start

    # ---- TIP-DECOMP ----
    start = time.time()
    S_tip = tip_decomp_deterministic(G)
    results["TIP-DECOMP"] = len(S_tip)
    results["runtime_TIP"] = time.time() - start

    # ---- TIP-DECOMP-R ----
    start = time.time()
    S_tipr = tip_decomp_random(G, seed=42)
    results["TIP-DECOMP-R"] = len(S_tipr)
    results["runtime_TIPR"] = time.time() - start

    # ---- RAND-P3 ----
    start = time.time()
    S_rand, k_rand = rand_p3(G, samples=500, velocity=2.0,
                             weighted=True, seed=42)
    results["RAND-P3"] = len(S_rand)
    results["runtime_RAND"] = time.time() - start

    return results


# ---------------------------------------------
# 3. Função para execução paralela
# ---------------------------------------------
def process_graph(path):
    """Executa todos os algoritmos em um único arquivo .tgf."""
    G = load_tgf(path)
    res = evaluate_algorithms(G)
    res["Graph"] = Path(path).name
    res["n"] = G.number_of_nodes()
    res["m"] = G.number_of_edges()
    return res


# ---------------------------------------------
# 4. Avaliação completa (com paralelismo)
# ---------------------------------------------
def run_full_evaluation(data_dir="data/tgf"):
    paths = sorted(Path(data_dir).glob("*.tgf"))
    n_cores = cpu_count()
    print(f"▶ Executando {len(paths)} grafos com {n_cores} núcleos...")

    results = []
    with Pool(n_cores) as pool:
        for res in tqdm(pool.imap_unordered(process_graph, paths), total=len(paths)):
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv("benchmark_tgf_results.csv", index=False)
    print("\n✅ Resultados salvos em benchmark_tgf_results.csv")

    return df


if __name__ == "__main__":
    run_full_evaluation()
