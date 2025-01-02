import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform
import torch.nn as nn

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cifar10_data(data_percentage=0.1, batch_size=64):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Determine the number of samples to use
    total_samples = len(trainset)
    samples_to_use = int(total_samples * data_percentage)
    
    # Randomly select a subset of data
    subset_indices = np.random.choice(total_samples, samples_to_use, replace=False)
    subset_data = torch.utils.data.Subset(trainset, subset_indices)
    
    # Create data loader
    dataloader = torch.utils.data.DataLoader(
        subset_data, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=2
    )
    
    return dataloader, trainset.classes

class MemoryEfficientFeatureExtractor:
    def __init__(self):
        resnet = torchvision.models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.features = self.features.to(device)
        self.features.eval()
    
    def extract_features(self, dataloader):
        features = []
        labels = []
        
        with torch.no_grad():
            for batch, (images, batch_labels) in enumerate(dataloader):
                images = images.to(device)
                batch_features = self.features(images).squeeze().cpu().numpy()
                
                features.append(batch_features)
                labels.extend(batch_labels.numpy())
                
                # Free up GPU memory
                torch.cuda.empty_cache()
        
        return np.vstack(features), np.array(labels)

class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphNeuralNetwork, self).__init__()
        # Reduced layer sizes to save memory
        self.conv1 = SAGEConv(num_features, 32)
        self.conv2 = SAGEConv(32, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def create_sparse_graph(features, k_neighbors=5):
    """
    Create a sparse graph by connecting each node to its k nearest neighbors
    """
    # Compute pairwise distances
    distances = pdist(features, metric='euclidean')
    dist_matrix = squareform(distances)
    
    # Create sparse adjacency matrix
    n = len(features)
    edge_index = []
    
    for i in range(n):
        # Find indices of k nearest neighbors
        neighbor_indices = np.argsort(dist_matrix[i])[1:k_neighbors+1]
        
        # Add bidirectional edges
        for neighbor in neighbor_indices:
            edge_index.extend([[i, neighbor], [neighbor, i]])
    
    return edge_index

def graph_learning(features, labels, num_classes, max_epochs=100):
    # Convert to PyTorch tensors
    x = torch.tensor(features, dtype=torch.float).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    
    # Create sparse edge index
    edge_index = create_sparse_graph(features)
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
    
    # Initialize model with smaller memory footprint
    model = GraphNeuralNetwork(features.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    
    train_losses = []
    test_losses = []
    
    print("Starting training...")
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        train_loss.backward()
        optimizer.step()
        
        # Evaluation phase
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
            pred = out.argmax(dim=1)
            test_acc = accuracy_score(
                data.y[data.test_mask].cpu().numpy(), 
                pred[data.test_mask].cpu().numpy()
            )
        
        # Record losses
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Test Loss = {test_loss.item():.4f}, Test Acc = {test_acc:.4f}")
        
        # Free up GPU memory
        torch.cuda.empty_cache()
    
    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig('loss_plot.png')
    plt.close()
    
    return test_acc, model

def main():
    # Key changes to reduce memory usage
    torch.cuda.empty_cache()
    
    # Get CIFAR10 data with batch loading
    dataloader, class_names = get_cifar10_data(data_percentage=0.5)
    print("Downloaded data")
    
    # Extract features using memory-efficient approach
    feature_extractor = MemoryEfficientFeatureExtractor()
    features, labels = feature_extractor.extract_features(dataloader)
    print("Feature extraction complete")
    
    # Perform graph learning
    test_accuracy, trained_model = graph_learning(features, labels, len(class_names))
    print(f"Graph Neural Network Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
