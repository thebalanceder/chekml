import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from gtn import GraphTransformerNeighbors, generate_graph_data, train_gtn, evaluate_gtn
from continuous_knn import ContinuousNearestNeighbors
from adaptive_knn import AdaptiveKNN
from hgnn import HGNN, generate_hypergraph_data, train_hgnn, evaluate_hgnn
from bayesian_knn import BayesianKNN
from li_nns import LearnedIndex, train_li_nns, build_faiss_index, predict_li_nns
from nne import NeuralEmbedding, train_nne, build_faiss_index as nne_build_faiss, predict_nne
from sl_knn import SelfLearningKNN
from s2_n2 import SelfSupervisedNeighborNetwork as S2N2, create_contrastive_dataloader as s2_create_dataloader, cluster_embeddings
from ml_knn import MetricLearningKNN
from sklearn.neighbors import KNeighborsClassifier
import time 

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Generate synthetic classification data for KNN-based models
    X, y = make_classification(n_samples=1000, n_features=5, n_classes=3, n_informative=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Demo GraphTransformerNeighbors
    print("=== GraphTransformerNeighbors Demo ===")
    gtn_data = generate_graph_data(num_nodes=100, num_edges=300, num_features=5, num_classes=3)
    gtn_model = GraphTransformerNeighbors(in_channels=5, hidden_channels=8, out_channels=3)
    train_time = train_gtn(gtn_model, gtn_data, epochs=100)
    _, accuracy, pred_time = evaluate_gtn(gtn_model, gtn_data)
    print(f"Training Time: {train_time:.6f} seconds")
    print(f"Prediction Time: {pred_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    # Demo ContinuousNearestNeighbors
    print("\n=== ContinuousNearestNeighbors Demo ===")
    cnn = ContinuousNearestNeighbors(kernel='gaussian', bandwidth=0.5)
    cnn.fit(X_train, y_train)
    y_pred = cnn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Training Time: {cnn.training_time:.6f} seconds")
    print(f"Prediction Time: {cnn.prediction_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    # Demo AdaptiveKNN
    print("\n=== AdaptiveKNN Demo ===")
    aknn = AdaptiveKNN(k_min=3, k_max=15, kernel='gaussian', bandwidth=0.5)
    aknn.fit(X_train, y_train)
    y_pred = aknn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Training Time: {aknn.training_time:.6f} seconds")
    print(f"Prediction Time: {aknn.prediction_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    # Demo HGNN
    print("\n=== HGNN Demo ===")
    hgnn_data = generate_hypergraph_data(num_nodes=10, num_edges=5, num_features=5, num_classes=3)
    hgnn_model = HGNN(in_channels=5, hidden_channels=10, out_channels=3)
    train_time = train_hgnn(hgnn_model, hgnn_data, epochs=100)
    _, accuracy, pred_time = evaluate_hgnn(hgnn_model, hgnn_data)
    print(f"Training Time: {train_time:.6f} seconds")
    print(f"Prediction Time: {pred_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    # Demo BayesianKNN
    print("\n=== BayesianKNN Demo ===")
    bknn = BayesianKNN(k=5)
    bknn.fit(X_train, y_train)
    y_pred = bknn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Training Time: {bknn.training_time:.6f} seconds")
    print(f"Prediction Time: {bknn.prediction_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    # Generate data for new models
    X_large, y_large = make_classification(n_samples=5000, n_features=10, n_classes=3, n_informative=3, random_state=42)
    X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(X_large, y_large, test_size=0.2, random_state=42)

    # Demo LearnedIndex (LI_NNS)
    print("\n=== LearnedIndex (LI_NNS) Demo ===")
    li_model = LearnedIndex(input_dim=10)
    train_time = train_li_nns(li_model, X_train_large, y_train_large, epochs=50)
    faiss_index = build_faiss_index(li_model, X_train_large)
    y_pred, pred_time = predict_li_nns(li_model, faiss_index, X_test_large, y_train_large, k=5)
    accuracy = np.mean(y_pred == y_test_large)
    print(f"Training Time: {train_time:.6f} seconds")
    print(f"Prediction Time: {pred_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    # Demo NeuralEmbedding (NNE)
    print("\n=== NeuralEmbedding (NNE) Demo ===")
    ne_model = NeuralEmbedding(input_dim=10)
    train_time = train_nne(ne_model, X_train_large, y_train_large, epochs=50)
    faiss_index = nne_build_faiss(ne_model, X_train_large)
    y_pred, pred_time = predict_nne(ne_model, faiss_index, X_test_large, y_train_large, k=5)
    accuracy = np.mean(y_pred == y_test_large)
    print(f"Training Time: {train_time:.6f} seconds")
    print(f"Prediction Time: {pred_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    # Demo SelfLearningKNN (SL_KNN)
    print("\n=== SelfLearningKNN (SL_KNN) Demo ===")
    X_labeled, X_unlabeled, y_labeled, _ = train_test_split(X, y, test_size=0.7, random_state=42)
    X_sl_train, X_sl_test, y_sl_train, y_sl_test = train_test_split(X_labeled, y_labeled, test_size=0.2, random_state=42)
    slknn = SelfLearningKNN(k=5, confidence_threshold=0.8, max_iterations=10)
    slknn.fit(X_sl_train, y_sl_train, X_unlabeled)
    y_pred = slknn.predict(X_sl_test)
    accuracy = np.mean(y_pred == y_sl_test)
    print(f"Training Time: {slknn.training_time:.6f} seconds")
    print(f"Prediction Time: {slknn.prediction_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

    # Demo SelfSupervisedNeighborNetwork (S2_N2) with KMeans
    print("\n=== SelfSupervisedNeighborNetwork (S2_N2) with KMeans Demo ===")
    X_s2 = np.random.rand(1000, 5).astype(np.float32)
    X_train_s2, X_test_s2, y_train_s2, y_test_s2 = train_test_split(X_s2, y, test_size=0.2, random_state=42)
    dataloader = s2_create_dataloader(X_train_s2, batch_size=32)
    s2_model = S2N2(model_type="mlp", input_dim=5)
    train_time = 0
    for epoch, loss in enumerate(s2_model.train(dataloader, epochs=10), 1):
        print(f"Epoch {epoch} | Loss: {loss:.4f}")
    train_time = s2_model.training_time
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_s2, dtype=torch.float32), torch.tensor(y_train_s2)), batch_size=32)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test_s2, dtype=torch.float32), torch.tensor(y_test_s2)), batch_size=32)
    train_embeddings = s2_model.get_embeddings(train_loader)
    test_embeddings = s2_model.get_embeddings(test_loader)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_embeddings, y_train_s2)
    start_time = time.time()
    y_pred = knn.predict(test_embeddings)
    pred_time = time.time() - start_time
    accuracy = np.mean(y_pred == y_test_s2)
    cluster_labels = cluster_embeddings(train_embeddings, method="kmeans", n_clusters=10)
    print(f"Training Time: {train_time:.6f} seconds")
    print(f"Prediction Time: {pred_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cluster Counts: {np.unique(cluster_labels, return_counts=True)}")

    # Demo MetricLearningKNN (ML_KNN)
    print("\n=== MetricLearningKNN (ML_KNN) Demo ===")
    mlknn = MetricLearningKNN(k=5, metric='mahalanobis', use_pca=True, pca_components=3)
    mlknn.fit(X_train, y_train)
    y_pred = mlknn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Training Time: {mlknn.training_time:.6f} seconds")
    print(f"Prediction Time: {mlknn.prediction_time:.6f} seconds")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
