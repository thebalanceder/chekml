import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        sim = torch.mm(z_i, z_j.T) / self.temperature
        labels = torch.arange(len(z_i), device=z_i.device)
        return F.cross_entropy(sim, labels)

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super(MLPEncoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        return self.model(x)

class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(CNNEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.fc_layers(x)

class SelfSupervisedNeighborNetwork:
    def __init__(self, model_type="mlp", input_dim=128, embedding_dim=128, temperature=0.5):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = MLPEncoder(input_dim, embedding_dim).to(self.device) if model_type == "mlp" else CNNEncoder(embedding_dim).to(self.device)
        self.loss_fn = NTXentLoss(temperature=temperature)
        self.optimizer = optim.Adam(self.encoder.parameters(), lr=0.001)
        self.training_time = 0

    def train(self, dataloader, epochs=10):
        self.encoder.train()
        start_time = time.time()
        for epoch in range(epochs):
            total_loss = 0
            for x1, x2, _ in dataloader:
                x1, x2 = x1.to(self.device), x2.to(self.device)
                z1 = self.encoder(x1)
                z2 = self.encoder(x2)
                loss = self.loss_fn(z1, z2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            yield total_loss / len(dataloader)
        self.training_time = time.time() - start_time

    def get_embeddings(self, dataloader):
        self.encoder.eval()
        embeddings = []
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)  # Handle both (x, y) and (x1, x2, y) formats
                z = self.encoder(x)
                embeddings.append(z.cpu().numpy())
        return np.vstack(embeddings)

def cluster_embeddings(embeddings, method="kmeans", n_clusters=10):
    if method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(embeddings)
    elif method == "dbscan":
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        return dbscan.fit_predict(embeddings)
    else:
        raise ValueError("Invalid clustering method! Choose 'kmeans' or 'dbscan'.")

def create_contrastive_dataloader(X, batch_size=32):
    X = torch.tensor(X, dtype=torch.float32)
    indices = np.random.randint(0, len(X), size=len(X))
    X2 = X[indices]
    dataset = TensorDataset(X, X2, torch.zeros(len(X)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
