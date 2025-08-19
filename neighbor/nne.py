import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import faiss
import numpy as np

class NeuralEmbedding(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=16):
        """
        Initialize Neural Embedding model.
        :param input_dim: Input feature dimension
        :param hidden_dim: Hidden layer dimension
        :param output_dim: Output embedding dimension
        """
        super(NeuralEmbedding, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def train_nne(model, X_train, y_train, epochs=50, batch_size=64, lr=0.01):
    """
    Train the Neural Embedding model.
    :param model: NeuralEmbedding instance
    :param X_train: Training data
    :param y_train: Training labels
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param lr: Learning rate
    :return: Training time (seconds)
    """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    for _ in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return time.time() - start_time

def build_faiss_index(model, X_train):
    """
    Build FAISS index using neural embeddings.
    :param model: Trained NeuralEmbedding model
    :param X_train: Training data
    :return: FAISS index
    """
    with torch.no_grad():
        X_train_embedded = model(torch.tensor(X_train, dtype=torch.float32)).numpy()
    faiss_index = faiss.IndexFlatL2(X_train_embedded.shape[1])
    faiss_index.add(X_train_embedded)
    return faiss_index

def predict_nne(model, faiss_index, X_test, y_train, k=5):
    """
    Predict using NNE.
    :param model: Trained NeuralEmbedding model
    :param faiss_index: FAISS index
    :param X_test: Test data
    :param y_train: Training labels
    :param k: Number of neighbors
    :return: Tuple of (predictions, prediction_time)
    """
    start_time = time.time()
    with torch.no_grad():
        X_test_embedded = model(torch.tensor(X_test, dtype=torch.float32)).numpy()
    D, I = faiss_index.search(X_test_embedded, k)
    y_pred = np.array([np.argmax(np.bincount(y_train[I[i]])) for i in range(len(I))])
    prediction_time = time.time() - start_time
    return y_pred, prediction_time
