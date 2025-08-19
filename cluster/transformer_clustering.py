import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class TransformerBasedClustering:
    def __init__(self, model_name="bert-base-uncased", num_clusters=2):
        """
        Initialize Transformer-Based Clustering.
        :param model_name: Name of the transformer model (e.g., 'bert-base-uncased')
        :param num_clusters: Number of clusters for KMeans
        """
        self.model_name = model_name
        self.num_clusters = num_clusters
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    
    def get_embedding(self, text):
        """
        Generate embedding for a single text.
        :param text: Input text string
        :return: CLS token embedding as numpy array
        """
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    def fit_predict(self, documents):
        """
        Generate embeddings and perform clustering.
        :param documents: List of text strings
        :return: Cluster labels
        """
        embeddings = np.array([self.get_embedding(doc) for doc in documents])
        labels = self.kmeans.fit_predict(embeddings)
        return labels, embeddings
    
    def visualize(self, embeddings, labels, title="Transformer-Based Clustering"):
        """
        Visualize clustering results using PCA.
        :param embeddings: Numpy array of embeddings
        :param labels: Cluster labels
        :param title: Plot title
        :return: PCA-transformed embeddings
        """
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        return reduced_embeddings
