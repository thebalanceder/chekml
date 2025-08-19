import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv
from torch_geometric.data import Data

class GraphTransformerNeighbors(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        """
        Initialize Graph Transformer Neighbors model.
        :param in_channels: Number of input features per node
        :param hidden_channels: Number of hidden features
        :param out_channels: Number of output classes
        :param heads: Number of attention heads for GATConv
        """
        super(GraphTransformerNeighbors, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = TransformerConv(hidden_channels * heads, out_channels, heads=1)

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

def generate_graph_data(num_nodes=100, num_edges=300, num_features=5, num_classes=3):
    """
    Generate synthetic graph data.
    :param num_nodes: Number of nodes
    :param num_edges: Number of edges
    :param num_features: Number of features per node
    :param num_classes: Number of classes
    :return: PyTorch Geometric Data object
    """
    x = torch.rand((num_nodes, num_features))
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, num_classes, (num_nodes,))
    return Data(x=x, edge_index=edge_index, y=y)

def train_gtn(model, data, epochs=100, lr=0.01):
    """
    Train the GTN model.
    :param model: GraphTransformerNeighbors instance
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

def evaluate_gtn(model, data):
    """
    Evaluate the GTN model.
    :param model: GraphTransformerNeighbors instance
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
