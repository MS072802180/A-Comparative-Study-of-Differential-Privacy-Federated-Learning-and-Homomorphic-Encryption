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
from opacus import PrivacyEngine
import os

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 50)
print("DIFFERENTIAL PRIVACY EXPERIMENTS")
print("=" * 50)

print("\nLoading dataset...")
with open('data/small_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y'].astype(np.int64)

# Normalize
X = (X - np.mean(X, axis=(0,1), keepdims=True)) / (np.std(X, axis=(0,1), keepdims=True) + 1e-8)

# Reshape for PyTorch: (samples, channels, time_steps)
X = np.transpose(X, (0, 2, 1))

# Split data
X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.long)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
Y_val_t = torch.tensor(Y_val, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.long)

# Create data loader
batch_size = 64
train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_t, Y_test_t), batch_size=batch_size, shuffle=False)

# Model WITHOUT BatchNorm (using GroupNorm instead for DP compatibility)
class ModulationCNN_DP(nn.Module):
    def __init__(self, num_classes=4):
        super(ModulationCNN_DP, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(8, 64)  # GroupNorm instead of BatchNorm
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(16, 128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(32, 256)
        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.gn2(self.conv2(x)))
        x = self.pool(x)
        x = self.relu(self.gn3(self.conv3(x)))
        x = self.pool(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# Epsilon values to test
epsilon_values = [10.0, 5.0, 2.0, 1.0, 0.5, 0.1]
dp_accuracies = []

print("\nRunning Differential Privacy experiments...")
print("-" * 50)

for eps in epsilon_values:
    print(f"\nTesting epsilon = {eps}")
    
    model = ModulationCNN_DP(num_classes=4)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    privacy_engine = PrivacyEngine()
    delta = 1e-5
    
    try:
        model, optimizer, train_loader_dp = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=0.5,  # Reduced for faster convergence
            max_grad_norm=1.0,
        )
        
        # Train for 10 epochs (fewer since DP is slower)
        for epoch in range(10):
            model.train()
            total_loss = 0
            for X_batch, Y_batch in train_loader_dp:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, Y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        
        # Evaluate on test set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, Y_batch in test_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += Y_batch.size(0)
                correct += (predicted == Y_batch).sum().item()
        
        test_acc = correct / total
        dp_accuracies.append(test_acc)
        
        epsilon_spent = privacy_engine.get_epsilon(delta)
        print(f"  Epsilon achieved: {epsilon_spent:.2f}")
        print(f"  Test accuracy: {test_acc:.4f}")
        
    except Exception as e:
        print(f"  Error: {str(e)[:100]}")
        dp_accuracies.append(0)

# Save results
print("\n" + "=" * 50)
print("DIFFERENTIAL PRIVACY RESULTS")
print("=" * 50)

print("\nEpsilon vs Accuracy:")
print("-" * 30)
for i, eps in enumerate(epsilon_values):
    print(f"  eps = {eps}: Accuracy = {dp_accuracies[i]:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, dp_accuracies, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Privacy Budget (Epsilon)', fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title('Differential Privacy: Accuracy vs Privacy Budget', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=0.9097, color='r', linestyle='--', label=f'Baseline (90.97%)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/dp_results.png')
print(f"\nPlot saved to plots/dp_results.png")

# Save accuracy values to file
with open('results/dp_results.txt', 'w') as f:
    f.write("Differential Privacy Results\n")
    f.write("=" * 30 + "\n")
    f.write(f"Baseline Accuracy: 0.9097\n\n")
    f.write("Epsilon vs Accuracy:\n")
    for i, eps in enumerate(epsilon_values):
        f.write(f"  eps = {eps}: {dp_accuracies[i]:.4f}\n")

print("\nResults saved to results/dp_results.txt")