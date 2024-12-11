import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data Preprocessing
def get_cifar10_data():
    # Normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load full CIFAR10 dataset (train + test)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Combine train and test datasets
    full_data = torch.utils.data.ConcatDataset([trainset, testset])
    
    return full_data, trainset.classes

class FeatureExtractor:
    def __init__(self):
        # Use a pre-trained ResNet for feature extraction
        resnet = torchvision.models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.features = self.features.to(device)
        self.features.eval()
    
    def extract_features(self, dataset):
        features = []
        labels = []
        with torch.no_grad():
            for img, label in dataset:
                img = img.to(device)
                feature = self.features(img.unsqueeze(0)).squeeze().cpu().numpy()
                features.append(feature)
                labels.append(label)
        return np.array(features), np.array(labels)


# Graph Construction
def construct_graph(features, similarity_threshold=0.8):
    # Compute pairwise distances
    distances = pdist(features, metric='cosine')
    dist_matrix = squareform(distances)
    
    # Create adjacency matrix based on similarity
    adj_matrix = (1 - dist_matrix) > similarity_threshold
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adj_matrix)
    
    # Visualize graph (sample visualization)
    plt.figure(figsize=(10, 10))
    nx.draw(G, node_size=10, node_color='blue', alpha=0.5)
    plt.title("CIFAR10 Similarity Graph")
    plt.savefig('cifar10_graph.png')
    plt.close()
    
    return G, adj_matrix

# Graph Neural Network
class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

 # Graph Learning Function with Loss Tracking
def graph_learning(features, labels, num_classes):
    # Convert to PyTorch tensors and move them to the device
    x = torch.tensor(features, dtype=torch.float).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    
    # Create edge index (fully connected graph)
    edge_index = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            edge_index.extend([[i, j], [j, i]])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)
    
    # Create PyTorch Geometric data
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Split data
    train_mask = torch.zeros(len(labels), dtype=torch.bool)
    test_mask = torch.zeros(len(labels), dtype=torch.bool)
    train_mask[:int(0.8*len(labels))] = True
    test_mask[int(0.8*len(labels)):] = True
    
    data.train_mask = train_mask.to(device)
    data.test_mask = test_mask.to(device)
    
    # Initialize and train GNN
    model = GraphNeuralNetwork(features.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    train_losses = []
    test_losses = []
    
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        
        # Calculate losses
        train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
        model.train()
        
        # Record losses
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Test Loss = {test_loss.item():.4f}")
    
    # Plot the training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_plot.png')
    plt.show()

    # Evaluate
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_acc = accuracy_score(
            data.y[data.test_mask].cpu().numpy(), 
            pred[data.test_mask].cpu().numpy()
        )
    
    return test_acc


# Subgraph Performance Analysis
def find_best_subgraph(features, labels, num_samples=100):
    # Randomly sample subgraphs and evaluate performance
    best_subgraph_acc = 0
    best_subgraph_indices = None
    
    for _ in range(100):  # Try 100 random subgraphs
        # Randomly select subset of nodes
        subgraph_indices = np.random.choice(
            len(features), 
            size=num_samples, 
            replace=False
        )
        
        # Extract subgraph features and labels
        subgraph_features = features[subgraph_indices]
        subgraph_labels = labels[subgraph_indices]
        
        # Perform graph learning on subgraph
        subgraph_acc = graph_learning(
            subgraph_features, 
            subgraph_labels, 
            len(np.unique(labels))
        )
        
        # Track best subgraph
        if subgraph_acc > best_subgraph_acc:
            best_subgraph_acc = subgraph_acc
            best_subgraph_indices = subgraph_indices
    
    return best_subgraph_indices, best_subgraph_acc

# Modified Data Preprocessing to use only 10% of the data
def get_cifar10_data(data_percentage=0.1):
    # Normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load full CIFAR10 dataset (train + test)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    # Combine train and test datasets
    full_data = torch.utils.data.ConcatDataset([trainset, testset])
    
    # Determine the number of samples to use
    total_samples = len(full_data)
    samples_to_use = int(total_samples * data_percentage)
    
    # Randomly select a subset of data
    subset_indices = np.random.choice(total_samples, samples_to_use, replace=False)
    subset_data = torch.utils.data.Subset(full_data, subset_indices)
    
    return subset_data, trainset.classes



def main():
    # Get 10% of CIFAR10 data
    subset_dataset, class_names = get_cifar10_data(data_percentage=0.1)
    print("Downloaded data")
    # Extract features
    feature_extractor = FeatureExtractor()
    features, labels = feature_extractor.extract_features(subset_dataset)
    print("Freature extraction complete ")
    # Construct graph
    graph, adj_matrix = construct_graph(features)
    print("Construced graph")
    # Perform graph learning
    test_accuracy = graph_learning(features, labels, len(class_names))
    print(f"Graph Neural Network Test Accuracy: {test_accuracy:.4f}")
    
    # Find best subgraph
    best_subgraph_indices, best_subgraph_acc = find_best_subgraph(features, labels)
    print(f"Best Subgraph Accuracy: {best_subgraph_acc:.4f}")
    print(f"Subgraph Size: {len(best_subgraph_indices)} nodes")

main()