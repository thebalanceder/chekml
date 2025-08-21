import numpy as np
from sklearn.datasets import make_moons, load_iris
import matplotlib.pyplot as plt
from transformer_clustering import TransformerBasedClustering
from self_evolving_clustering import SelfEvolvingClustering
from adaptive_graph_clustering import AdaptiveGraphClustering
from ddbc_clustering import DDBC

def main():
    # Demo TransformerBasedClustering
    print("=== TransformerBasedClustering Demo ===")
    documents = [
        "Machine learning is amazing!",
        "Deep learning drives AI progress.",
        "Transformers are powerful models.",
        "The economy is experiencing inflation.",
        "Stock markets fluctuate daily.",
        "Investing in crypto is risky.",
    ]
    tbc = TransformerBasedClustering(model_name="bert-base-uncased", num_clusters=2)
    labels, embeddings = tbc.fit_predict(documents)
    reduced_embeddings = tbc.visualize(embeddings, labels)
    print("Transformer-Based Clustering Labels:", labels)

    # Demo SelfEvolvingClustering
    print("\n=== SelfEvolvingClustering Demo ===")
    np.random.seed(42)
    data = np.random.rand(100, 2)
    sec = SelfEvolvingClustering(distance_threshold=0.2, learning_rate=0.05, merge_threshold=0.1)
    sec.fit(data)
    labels = sec.predict(data)
    print(f"Self-Evolving Clustering: {len(sec.clusters)} clusters")

    # Demo AdaptiveGraphClustering
    print("\n=== AdaptiveGraphClustering Demo ===")
    iris = load_iris()
    data = iris.data
    true_labels = iris.target
    agc = AdaptiveGraphClustering(k=5)
    labels = agc.fit_predict(data)
    ari, nmi = agc.evaluate(true_labels, labels)
    print("Adaptive Graph Clustering Labels:", labels[:10])
    print(f"ARI: {ari:.4f}, NMI: {nmi:.4f}")

    # Demo DDBC
    print("\n=== DDBC Demo ===")
    X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)
    ddbc = DDBC(bandwidth=0.2, reachability_threshold=0.5)
    soft_labels = ddbc.fit_predict(X)
    print("DDBC Soft Labels (first 10):", soft_labels[:10])

if __name__ == "__main__":
    main()
