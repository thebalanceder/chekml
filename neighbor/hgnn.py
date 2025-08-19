import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_geometric.data import Data

class HGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        """
        Initialize Hypergraph Neural Network.
        :param in_channels: Number of input features per node
        :param hidden_channels: Number of hidden features
        :param out_channels: Number of output classes
        """
        super(HGNN, self).__init__()
        self.conv1 = HypergraphConv(in_channels, hidden_channels)
        self.conv2 = HypergraphConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        """
        Forward pass of the model.
        :param x: Node feature tensor
        :param edge_index: Edge index tensor
        :return: Log softmax probabilities
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def generate_hypergraph_data(num_nodes=10, num_edges=5, num_features=5, num_classes=3):
    """
    Generate synthetic hypergraph data.
    :param num_nodes: Number of nodes
    :param num_edges: Number of hyperedges
    :param num_features: Number of features per node
    :param num_classes: Number of classes
    :return: PyTorch Geometric Data object
    """
    x = torch.rand((num_nodes, num_features))
    edge_index = torch.randint(0, num_nodes, (2, num_edges * num_nodes // 2))
    y = torch.randint(0, num_classes, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)

def train_hgnn(model, data, epochs=100, lr=0.01):
    """
    Train the HGNN model.
    :param model: HGNN instance
    :param data: PyTorch Geometric Data object
    :param epochs: Number of training epochs
    :param lr: Learning rate
    :return: Training time (seconds)
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.NLLLoss()
    start_time = time.time()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
    return time.time() - start_time

def evaluate_hgnn(model, data):
    """
    Evaluate the HGNN model.
    :param model: HGNN instance
    :param data: PyTorch Geometric Data object
    :return: Tuple of (predictions, accuracy, prediction_time)
    """
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        predictions = model(data.x, data.edge_index).argmax(dim=1)
    prediction_time = time.time() - start_time
    accuracy = (predictions == data.y).float().mean().item()
    return predictions, accuracy, prediction_time
