import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import flwr as fl
from collections import OrderedDict
import os

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 50)
print("FEDERATED LEARNING EXPERIMENTS")
print("=" * 50)

print("\nLoading dataset...")
with open('data/small_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y'].astype(np.int64)

# Normalize
X = (X - np.mean(X, axis=(0,1), keepdims=True)) / (np.std(X, axis=(0,1), keepdims=True) + 1e-8)

# Reshape
X = np.transpose(X, (0, 2, 1))

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Convert to tensors
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.long)

# Model definition (same as before)
class ModulationCNN_FL(nn.Module):
    def __init__(self, num_classes=4):
        super(ModulationCNN_FL, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Helper function to get model parameters
def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

# Helper function to set model parameters
def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# Create client datasets (non-IID distribution)
def create_client_datasets(X_train, Y_train, num_clients=5):
    samples_per_client = len(X_train) // num_clients
    client_datasets = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X_train)
        
        X_client = torch.tensor(X_train[start_idx:end_idx], dtype=torch.float32)
        Y_client = torch.tensor(Y_train[start_idx:end_idx], dtype=torch.long)
        
        client_datasets.append(TensorDataset(X_client, Y_client))
    
    return client_datasets

# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader):
        self.model = model
        self.train_loader = train_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def get_parameters(self, config):
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.train()
        
        for epoch in range(3):  # 3 local epochs
            for X_batch, Y_batch in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, Y_batch)
                loss.backward()
                self.optimizer.step()
        
        return get_parameters(self.model), len(self.train_loader.dataset), {}
    
    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
        
        test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=64)
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += Y_batch.size(0)
                correct += (predicted == Y_batch).sum().item()
        
        return float(correct / total), len(test_loader.dataset), {}

# Run FL with different numbers of clients
client_counts = [2, 5, 10, 20]
fl_accuracies = []

print("\nRunning Federated Learning experiments...")
print("-" * 50)

for num_clients in client_counts:
    print(f"\nNumber of clients: {num_clients}")
    
    # Create client datasets
    client_datasets = create_client_datasets(X_train, Y_train, num_clients)
    
    # Create clients
    clients = []
    for i, dataset in enumerate(client_datasets):
        train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        model = ModulationCNN_FL(num_classes=4)
        clients.append(FlowerClient(model, train_loader).to_client())
    
    # Start FL server (using in-memory simulation)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=num_clients,
        min_available_clients=num_clients,
    )
    
    # Simulate FL
    history = fl.simulation.start_simulation(
        client_fn=lambda cid: clients[int(cid)],
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )
    
    # Get final accuracy
    final_accuracy = history.metrics_centralized['centralized_accuracy'][-1][1] if history.metrics_centralized else 0
    fl_accuracies.append(final_accuracy)
    print(f"  Final test accuracy: {final_accuracy:.4f}")

# Also run centralized baseline for comparison
print("\n" + "=" * 50)
print("FEDERATED LEARNING RESULTS")
print("=" * 50)

print("\nCentralized training baseline: 0.9097 (from earlier)")

print("\nClients vs Accuracy:")
print("-" * 25)
for i, num_clients in enumerate(client_counts):
    print(f"  {num_clients} clients: {fl_accuracies[i]:.4f}")

# Plot
plt.figure(figsize=(10, 6))
plt.plot(client_counts, fl_accuracies, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clients', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Federated Learning: Accuracy vs Number of Clients', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=0.9097, color='r', linestyle='--', label=f'Centralized (90.97%)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/fl_results.png')
print(f"\nPlot saved to plots/fl_results.png")

# Save results
with open('results/fl_results.txt', 'w') as f:
    f.write("Federated Learning Results\n")
    f.write("=" * 30 + "\n")
    f.write(f"Centralized Baseline: 0.9097\n\n")
    f.write("Number of Clients vs Accuracy:\n")
    for i, num_clients in enumerate(client_counts):
        f.write(f"  {num_clients} clients: {fl_accuracies[i]:.4f}\n")

print("\nResults saved to results/fl_results.txt")