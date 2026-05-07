import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import os

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 50)
print("FEDERATED LEARNING - FIXED VERSION")
print("=" * 50)

# Load dataset
print("\nLoading dataset...")
with open('data/small_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y'].astype(np.int64)

# Normalize
X = (X - np.mean(X, axis=(0,1), keepdims=True)) / (np.std(X, axis=(0,1), keepdims=True) + 1e-8)

# Reshape for PyTorch
X = np.transpose(X, (0, 2, 1))

# Split into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.long)

test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=64, shuffle=False)

# Model definition (same as baseline)
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

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, Y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += Y_batch.size(0)
            correct += (predicted == Y_batch).sum().item()
    return correct / total

def train_local(model, train_loader, epochs=3, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for X_batch, Y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, Y_batch)
            loss.backward()
            optimizer.step()
    return model

def federated_average(global_model, client_models):
    """Average client models into global model"""
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        if global_dict[key].dtype == torch.float32 or global_dict[key].dtype == torch.float64:
            avg = torch.zeros_like(global_dict[key], dtype=torch.float32)
            for client_model in client_models:
                avg += client_model.state_dict()[key].float()
            avg /= len(client_models)
            global_dict[key] = avg.to(global_dict[key].dtype)
    
    global_model.load_state_dict(global_dict)
    return global_model

# First, train centralized baseline
print("\nTraining centralized baseline...")
central_model = ModulationCNN_FL(num_classes=4)
train_loader_central = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=64, shuffle=True)
central_model = train_local(central_model, train_loader_central, epochs=10)
central_acc = evaluate_model(central_model, test_loader)
print(f"Centralized baseline accuracy: {central_acc:.4f}")

# FL experiment
client_counts = [2, 5, 10, 20]
fl_accuracies = []
round_accuracies_all = []

print("\nRunning Federated Learning experiments...")
print("-" * 50)

for num_clients in client_counts:
    print(f"\nNumber of clients: {num_clients}")
    
    # Split data into client partitions
    samples_per_client = len(X_train) // num_clients
    client_loaders = []
    
    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else len(X_train)
        
        X_client = X_train_t[start_idx:end_idx]
        Y_client = Y_train_t[start_idx:end_idx]
        client_loader = DataLoader(TensorDataset(X_client, Y_client), batch_size=32, shuffle=True)
        client_loaders.append(client_loader)
    
    # Initialize global model
    global_model = ModulationCNN_FL(num_classes=4)
    
    # Federated averaging rounds
    num_rounds = 10
    round_accuracies = []
    
    for round_num in range(num_rounds):
        client_models = []
        
        # Train each client locally
        for client_loader in client_loaders:
            local_model = copy.deepcopy(global_model)
            local_model = train_local(local_model, client_loader, epochs=3)
            client_models.append(local_model)
        
        # Average model parameters (FedAvg)
        global_model = federated_average(global_model, client_models)
        
        # Evaluate after each round
        acc = evaluate_model(global_model, test_loader)
        round_accuracies.append(acc)
        
        if (round_num + 1) % 5 == 0:
            print(f"  Round {round_num+1}: accuracy = {acc:.4f}")
    
    final_acc = round_accuracies[-1]
    fl_accuracies.append(final_acc)
    round_accuracies_all.append(round_accuracies)
    print(f"  Final accuracy after {num_rounds} rounds: {final_acc:.4f}")

# Results
print("\n" + "=" * 50)
print("FEDERATED LEARNING RESULTS")
print("=" * 50)
print(f"\nCentralized baseline: {central_acc:.4f}")
print("\nNumber of Clients vs Accuracy:")
print("-" * 30)
for i, num_clients in enumerate(client_counts):
    print(f"  {num_clients} clients: {fl_accuracies[i]:.4f}")

# Save results to file
with open('results/federated_learning_results.txt', 'w') as f:
    f.write("Federated Learning Results\n")
    f.write("=" * 30 + "\n")
    f.write(f"Centralized Baseline: {central_acc:.4f}\n\n")
    f.write("Number of Clients vs Accuracy:\n")
    for i, num_clients in enumerate(client_counts):
        f.write(f"  {num_clients} clients: {fl_accuracies[i]:.4f}\n")
    f.write("\n\nDetailed round-by-round accuracy:\n")
    for i, num_clients in enumerate(client_counts):
        f.write(f"\n{num_clients} clients:\n")
        for round_num, acc in enumerate(round_accuracies_all[i]):
            f.write(f"  Round {round_num+1}: {acc:.4f}\n")

print("\nResults saved to results/federated_learning_results.txt")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(client_counts, fl_accuracies, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clients', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Federated Learning: Accuracy vs Number of Clients', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=central_acc, color='r', linestyle='--', label=f'Centralized ({central_acc:.4f})')
plt.legend()
plt.tight_layout()
plt.savefig('plots/federated_learning_results.png')
print(f"Plot saved to plots/federated_learning_results.png")

# Also plot round-by-round convergence for each client count
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, num_clients in enumerate(client_counts):
    ax = axes[idx]
    ax.plot(range(1, 11), round_accuracies_all[idx], 'go-', linewidth=2, markersize=6)
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'{num_clients} Clients')
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=central_acc, color='r', linestyle='--', label=f'Centralized ({central_acc:.4f})')
    ax.legend()

plt.tight_layout()
plt.savefig('plots/federated_learning_convergence.png')
print(f"Convergence plot saved to plots/federated_learning_convergence.png")

print("\n" + "=" * 50)
print("FEDERATED LEARNING COMPLETE")
print("=" * 50)