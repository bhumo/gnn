import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from scipy.spatial.distance import pdist, squareform

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_cifar10_data(data_percentage=0.1, batch_size=64):
    """
    Load CIFAR-10 dataset with optional data percentage and data augmentation
    """
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
    """
    Extract features using a pre-trained ResNet18 model
    """
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
    """
    Graph Neural Network with GraphSAGE Convolutions
    """
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

def create_graph_with_similarity(features, similarity_threshold=0.8):
    """
    Create a graph based on feature similarity
    """
    # Normalize features to compute cosine similarity
    normalized_features = features / np.linalg.norm(features, axis=1)[:, np.newaxis]
    similarity_matrix = np.dot(normalized_features, normalized_features.T)
    
    # Create sparse adjacency matrix based on similarity threshold
    n = len(features)
    edge_index = []
    
    for i in range(n):
        for j in range(i+1, n):
            # Create an edge if similarity is above threshold
            if similarity_matrix[i, j] > similarity_threshold:
                edge_index.extend([[i, j], [j, i]])
    
    return edge_index, similarity_matrix

def graph_learning(features, labels, num_classes, max_epochs=100):
    """
    Perform graph-based learning on the features
    """
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
    plt.savefig('loss_plot_l2.png')
    plt.close()
    
    return test_acc, model


def visualize_graph(features, edge_index, labels, class_names, save_path='similarity_graph_without_pca.png'):
    """
    Visualize the graph using NetworkX without PCA
    """
    if isinstance(edge_index, list):
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    # Create NetworkX graph
    G = nx.Graph()
    
    # Add nodes with labels and class names
    for i in range(len(features)):
        G.add_node(i, label=labels[i], class_name=class_names[labels[i]])
    
    # Add edges based on edge_index
    edges = [(int(edge_index[0][i]), int(edge_index[1][i])) for i in range(edge_index.shape[1])]
    G.add_edges_from(edges)
    
    # Prepare visualization
    plt.figure(figsize=(20, 20))
    
    # Use a circular layout for the graph to avoid dimensionality reduction
    node_positions = nx.circular_layout(G)  # or any layout you prefer
    
    # Color mapping for classes
    unique_classes = np.unique(labels)
    color_map = plt.cm.rainbow(np.linspace(0, 1, len(unique_classes)))
    
    # Map colors to classes
    node_colors = [color_map[np.where(unique_classes == label)[0][0]] for label in labels]
    
    # Draw the graph using NetworkX's draw function
    nx.draw(
        G, 
        pos=node_positions, 
        node_color=node_colors, 
        node_size=50, 
        with_labels=False, 
        edge_color='gray', 
        alpha=0.7
    )
    
    # Create a custom legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   label=class_names[cls], 
                                   markerfacecolor=color_map[i], 
                                   markersize=10) 
                       for i, cls in enumerate(unique_classes)]
    
    plt.legend(handles=legend_elements, title='Classes', loc='best')
    plt.title('Similarity Graph of CIFAR-10 Features (No PCA)')
    
    # Save the graph
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    # Print graph statistics
    print(f"Graph Statistics:")
    print(f"Number of Nodes: {G.number_of_nodes()}")
    print(f"Number of Edges: {G.number_of_edges()}")
    print(f"Average Node Degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")


def sample_subgraph(features, labels, edge_index, sampling_strategy='degree', sample_ratio=0.5):
    """
    Sample a sub-graph from the original graph based on different strategies
    """
    # Convert edge_index to networkx graph for analysis
    G = nx.Graph()
    for i, j in zip(edge_index[0], edge_index[1]):
        G.add_edge(int(i), int(j))
    
    # Node selection strategies
    if sampling_strategy == 'degree':
        # Sample nodes with highest degree
        degrees = dict(G.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)
    elif sampling_strategy == 'pagerank':
        # Sample nodes based on PageRank centrality
        pagerank = nx.pagerank(G)
        top_nodes = sorted(pagerank, key=pagerank.get, reverse=True)
    else:
        # Random sampling
        top_nodes = list(G.nodes())
        np.random.shuffle(top_nodes)
    
    # Select subset of nodes
    num_samples = int(len(top_nodes) * sample_ratio)
    selected_nodes = top_nodes[:num_samples]
    
    # Filter features and labels
    mask = np.isin(np.arange(len(features)), selected_nodes)
    sub_features = features[mask]
    sub_labels = labels[mask]
    
    # Reconstruct edge_index for subgraph
    node_mapping = {old: new for new, old in enumerate(selected_nodes)}
    sub_edge_index = []
    for i, j in zip(edge_index[0], edge_index[1]):
        if int(i) in selected_nodes and int(j) in selected_nodes:
            sub_edge_index.append([node_mapping[int(i)], node_mapping[int(j)]])
    
    return (
        sub_features, 
        sub_labels, 
        torch.tensor(sub_edge_index, dtype=torch.long).t().contiguous()
    )
# (Previous imports and entire previous code remain the same)

def evaluate_subgraph_performance(
    features, 
    labels, 
    edge_index, 
    num_classes, 
    sampling_strategies=['degree', 'random', 'pagerank'],
    sample_ratios=[0.2, 0.3, 0.4, 0.5]
):
    """
    Evaluate performance of sub-graphs using different sampling strategies
    """
    results = {}
    
    for strategy in sampling_strategies:
        strategy_results = []
        for ratio in sample_ratios:
            # Sample subgraph
            sub_features, sub_labels, sub_edge_index = sample_subgraph(
                features, labels, edge_index, 
                sampling_strategy=strategy, 
                sample_ratio=ratio
            )
            
            # Convert to PyTorch tensors
            x = torch.tensor(sub_features, dtype=torch.float).to(device)
            y = torch.tensor(sub_labels, dtype=torch.long).to(device)
          
            # Split subgraph data
            train_mask = torch.zeros(len(sub_labels), dtype=torch.bool)
            test_mask = torch.zeros(len(sub_labels), dtype=torch.bool)
            train_mask[:int(0.8*len(sub_labels))] = True
            test_mask[int(0.8*len(sub_labels)):] = True
            
            # Create PyTorch Geometric data
            data = Data(x=x, edge_index=sub_edge_index, y=y)
            data.train_mask = train_mask.to(device)
            data.test_mask = test_mask.to(device)
            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            # Initialize and train model
            model = GraphNeuralNetwork(sub_features.shape[1], num_classes).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
            
            best_test_acc = 0
            for _ in range(50):  # Reduced epochs for subgraph
                model.train()
                optimizer.zero_grad()
    
                out = model(data.x, data.edge_index)
                train_loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
                train_loss.backward()
                optimizer.step()
                # print("Evalaution")
                # Evaluation
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    test_loss = F.nll_loss(out[data.test_mask], data.y[data.test_mask])
                    pred = out.argmax(dim=1)
                    test_acc = accuracy_score(
                        data.y[data.test_mask].cpu().numpy(), 
                        pred[data.test_mask].cpu().numpy()
                    )
                    best_test_acc = max(best_test_acc, test_acc)
            
            strategy_results.append({
                'sampling_strategy': strategy,
                'sample_ratio': ratio,
                'test_accuracy': best_test_acc
            })
        
        results[strategy] = strategy_results
    
    # Visualization of results
    plt.figure(figsize=(12, 6))
    for strategy in sampling_strategies:
        accuracies = [r['test_accuracy'] for r in results[strategy]]
        plt.plot(sample_ratios, accuracies, marker='o', label=strategy)
    
    plt.xlabel('Sample Ratio')
    plt.ylabel('Test Accuracy')
    plt.title('Subgraph Performance Across Sampling Strategies')
    plt.legend()
    plt.grid(True)
    plt.savefig('subgraph_performance_l2.png')
    plt.close()
    
    return results

def main():
    """
    Main function to run the entire graph-based machine learning pipeline
    """
    # Key changes to reduce memory usage
    torch.cuda.empty_cache()
    
    # Get CIFAR10 data with batch loading
    dataloader, class_names = get_cifar10_data(data_percentage=0.1)
    print("Downloaded data")
    
    # Extract features using memory-efficient approach
    feature_extractor = MemoryEfficientFeatureExtractor()
    features, labels = feature_extractor.extract_features(dataloader)
    print("Feature extraction complete")
    similarity_edge_index, _ = create_graph_with_similarity(features, similarity_threshold=0.8)
    visualize_graph(features, similarity_edge_index, labels, class_names)
    
    # Create original graph edge index
    edge_index = create_sparse_graph(features)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Original graph learning
    test_accuracy, trained_model = graph_learning(features, labels, len(class_names))
    print(f"Original Graph Neural Network Test Accuracy: {test_accuracy:.4f}")
    
    # Create and visualize similarity graph
   # similarity_edge_index, _ = create_graph_with_similarity(features, similarity_threshold=0.8)
   # visualize_graph(features, similarity_edge_index, labels, class_names)
    
    # Evaluate subgraph performance
    subgraph_results = evaluate_subgraph_performance(
        features, labels, edge_index_tensor.numpy(), len(class_names)
    )
    
    # Print and analyze results
    print("\nSubgraph Performance Results:")
    for strategy, results in subgraph_results.items():
        print(f"\n{strategy.capitalize()} Sampling Strategy:")
        for result in results:
            print(f"Sample Ratio: {result['sample_ratio']}, Test Accuracy: {result['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()

