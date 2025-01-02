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

def get_cifar10_data(data_percentage=0.2):
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
    
    return subset_data, trainset.classes

class FeatureExtractor:
    def __init__(self):
        resnet = torchvision.models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.features = self.features.to(device)
        self.features.eval()
    
    def extract_features(self, dataset):
        features = []
        labels = []
        with torch.no_grad():
            for img, label in dataset:
                img = img.unsqueeze(0).to(device)
                feature = self.features(img).squeeze().flatten().cpu().numpy()
                features.append(feature)
                labels.append(label)
        return np.array(features), np.array(labels)

class GraphNeuralNetwork(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = SAGEConv(num_features, 64)
        self.conv2 = SAGEConv(64, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def construct_graph(features, similarity_threshold=0.8):
    # Compute pairwise distances
    distances = pdist(features, metric='cosine')
    dist_matrix = squareform(distances)
    
    # Create adjacency matrix based on similarity
    adj_matrix = (1 - dist_matrix) > similarity_threshold
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adj_matrix)
    
    return G, adj_matrix

def graph_learning(features, labels, num_classes, max_epochs=200, patience=20):
    # Convert to PyTorch tensors
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
    
    # Initialize model
    model = GraphNeuralNetwork(features.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Early stopping variables
    best_test_loss = float('inf')
    epochs_no_improve = 0
    best_model = None
    
    train_losses = []
    test_losses = []
    
    print("Starting training...")
    
    for epoch in range(max_epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        
        try:
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
            
            # Early stopping logic
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                epochs_no_improve = 0
                best_model = model.state_dict().copy()
            else:
                epochs_no_improve += 1
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss.item():.4f}, Test Loss = {test_loss.item():.4f}, Test Acc = {test_acc:.4f}")
            
            # Early stopping
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        except Exception as e:
            print(f"Error in epoch {epoch}: {e}")
            break
    
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
    
    # Restore best model if available
    if best_model:
        model.load_state_dict(best_model)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        test_acc = accuracy_score(
            data.y[data.test_mask].cpu().numpy(), 
            pred[data.test_mask].cpu().numpy()
        )
    
    return test_acc, model

def main():
    # Get CIFAR10 data
    subset_dataset, class_names = get_cifar10_data(data_percentage=0.1)
    print("Downloaded data")
    
    # Extract features
    feature_extractor = FeatureExtractor()
    features, labels = feature_extractor.extract_features(subset_dataset)
    print("Feature extraction complete")
    
    # Perform graph learning
    test_accuracy, trained_model = graph_learning(features, labels, len(class_names))
    print(f"Graph Neural Network Test Accuracy: {test_accuracy:.4f}")

if __name__ == "__main__":
    main()
