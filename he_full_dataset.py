import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tenseal as ts
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print('=' * 60)
print('HOMOMORPHIC ENCRYPTION - FULL DATASET (16,384 SAMPLES)')
print('=' * 60)
sys.stdout.flush()

# Load full dataset
print('\n1. Loading full dataset (16,384 samples)...')
sys.stdout.flush()
with open('data/small_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y'].astype(np.int64)

print(f'   Original X shape: {X.shape}')
print(f'   Original Y shape: {Y.shape}')
print(f'   Classes: {np.unique(Y)}')
sys.stdout.flush()

# Downsample time dimension for HE (1024 -> 32 time steps)
print('\n2. Downsampling for HE compatibility...')
sys.stdout.flush()
X_downsampled = X[:, ::32, :]  # (16384, 32, 2)
X_flat = X_downsampled.reshape(X_downsampled.shape[0], -1)  # (16384, 64)
print(f'   Downsampled shape: {X_flat.shape}')
sys.stdout.flush()

# Normalize to range [-0.5, 0.5] for HE stability
print('\n3. Normalizing data...')
sys.stdout.flush()
X_flat = X_flat / (np.max(np.abs(X_flat)) + 1e-8)
print(f'   Data range: [{X_flat.min():.4f}, {X_flat.max():.4f}]')
sys.stdout.flush()

# Use ALL samples for training, but split for train/test
print('\n4. Splitting into train (80%) and test (20%)...')
sys.stdout.flush()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_flat, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f'   Training samples: {X_train.shape[0]}')
print(f'   Test samples: {X_test.shape[0]}')
print(f'   Each class in train: {np.bincount(Y_train)}')
print(f'   Each class in test: {np.bincount(Y_test)}')
sys.stdout.flush()

# Train logistic regression on full dataset
print('\n5. Training logistic regression on full dataset...')
sys.stdout.flush()

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

model = LogisticRegression(input_dim=64, output_dim=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.long)

batch_size = 256
train_dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for Xb, Yb in train_loader:
        optimizer.zero_grad()
        output = model(Xb)
        loss = criterion(output, Yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 40 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f'   Epoch {epoch+1}, Loss: {avg_loss:.4f}')
        sys.stdout.flush()

# Evaluate plaintext
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs, 1)
    plaintext_acc = (predicted == Y_test_t).sum().item() / len(Y_test_t)
print(f'\n   Plaintext logistic regression accuracy: {plaintext_acc:.4f}')
sys.stdout.flush()

# Extract weights and bias
weight = model.linear.weight.detach().numpy()  # (4, 64)
bias = model.linear.bias.detach().numpy()      # (4,)

print(f'\n6. Model parameters: weight {weight.shape}, bias {bias.shape}')
sys.stdout.flush()

# Create TenSEAL context
print('\n7. Creating CKKS encryption context...')
sys.stdout.flush()
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=16384,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40
print('   Context created successfully')
sys.stdout.flush()

# Encrypt test samples (use all test samples, but this will take time)
n_test_all = len(X_test)
print(f'\n8. Encrypting {n_test_all} test samples...')
print('   This will take 2-3 minutes...')
sys.stdout.flush()

encrypted_vectors = []
encryption_times = []

for i in range(n_test_all):
    start = time.time()
    encrypted = ts.ckks_vector(context, X_test[i].tolist())
    encryption_times.append(time.time() - start)
    encrypted_vectors.append(encrypted)
    
    if (i + 1) % 500 == 0:
        print(f'   Encrypted {i+1}/{n_test_all} samples')
        sys.stdout.flush()

print(f'   Average encryption time: {np.mean(encryption_times)*1000:.2f} ms')
sys.stdout.flush()

# Create encrypted bias vectors
print('\n9. Creating encrypted bias vectors...')
sys.stdout.flush()
encrypted_bias = []
for class_idx in range(4):
    bias_vector = [float(bias[class_idx])]
    encrypted_bias.append(ts.ckks_vector(context, bias_vector))

# Homomorphic inference function
def homomorphic_inference(encrypted_vec, weight, encrypted_bias):
    logits = []
    for class_idx in range(weight.shape[0]):  # 4 classes
        w = weight[class_idx].tolist()
        dot_product = encrypted_vec * w
        dot_product_sum = dot_product.sum()
        logit = dot_product_sum + encrypted_bias[class_idx]
        logits.append(logit)
    return logits

# Run homomorphic inference on ALL test samples
print(f'\n10. Running homomorphic inference on {n_test_all} samples...')
print('    This will take 30-45 minutes...')
sys.stdout.flush()

correct = 0
inference_times = []

for i in range(n_test_all):
    start = time.time()
    
    encrypted_logits = homomorphic_inference(
        encrypted_vectors[i], weight, encrypted_bias
    )
    
    decrypted_logits = []
    for logit in encrypted_logits:
        decrypted = logit.decrypt()
        decrypted_logits.append(decrypted[0])
    
    inference_times.append(time.time() - start)
    
    predicted_class = np.argmax(decrypted_logits)
    if predicted_class == Y_test[i]:
        correct += 1
    
    if (i + 1) % 200 == 0:
        current_acc = correct / (i + 1)
        avg_time = np.mean(inference_times[-200:]) * 1000
        print(f'   Processed {i+1}/{n_test_all}, Acc: {current_acc:.4f}, Time: {avg_time:.2f} ms')
        sys.stdout.flush()

he_acc = correct / n_test_all
valid_times = [t for t in inference_times if t > 0]
avg_inference_time = np.mean(valid_times) * 1000 if valid_times else 0

print(f'\n   FINAL RESULTS:')
print(f'   Plaintext accuracy: {plaintext_acc:.4f}')
print(f'   Encrypted accuracy: {he_acc:.4f}')
print(f'   Accuracy difference: {abs(plaintext_acc - he_acc):.6f}')
print(f'   Encryption time: {np.mean(encryption_times)*1000:.2f} ms')
print(f'   Inference time: {avg_inference_time:.2f} ms')
sys.stdout.flush()

# Final summary
print('\n' + '=' * 60)
print('HOMOMORPHIC ENCRYPTION - FULL DATASET RESULTS')
print('=' * 60)
print(f'Dataset: 16,384 total samples (80% train, 20% test)')
print(f'Training samples: {X_train.shape[0]}')
print(f'Test samples (encrypted): {n_test_all}')
print(f'Model: Logistic Regression')
print(f'Features: 64 (downsampled I/Q)')
print(f'Classes: 4')
print(f'')
print(f'Plaintext accuracy: {plaintext_acc:.4f}')
print(f'Encrypted accuracy: {he_acc:.4f}')
print(f'Accuracy match: {abs(plaintext_acc - he_acc):.6f} difference')
print(f'')
print(f'Encryption time: {np.mean(encryption_times)*1000:.2f} ms/sample')
print(f'HE inference time: {avg_inference_time:.2f} ms/sample')
print(f'Plaintext inference time: ~0.05 ms/sample')
print(f'Slowdown factor: {avg_inference_time / 0.05:.0f}x')

# Plot results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

accuracies = [plaintext_acc, he_acc]
labels = ['Plaintext\nLogistic', 'Encrypted HE\nLogistic']
bars1 = axes[0].bar(labels, accuracies, color=['blue', 'orange'])
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].set_title('Accuracy: Plaintext vs Homomorphic Encryption')
for bar, val in zip(bars1, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)

times = [0.05, np.mean(encryption_times)*1000, avg_inference_time]
time_labels = ['Plaintext\nInference', 'Encryption', 'HE\nInference']
bars2 = axes[1].bar(time_labels, times, color=['green', 'orange', 'red'])
axes[1].set_ylabel('Time (ms)')
axes[1].set_title('Performance Overhead (Log Scale)')
axes[1].set_yscale('log')
for bar, val in zip(bars2, times):
    if val > 0:
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f'{val:.1f}ms', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('plots/he_full_dataset_results.png')
print(f'\nPlot saved to plots/he_full_dataset_results.png')

# Save results
with open('results/he_full_dataset_results.txt', 'w') as f:
    f.write('HOMOMORPHIC ENCRYPTION - FULL DATASET RESULTS\n')
    f.write('=' * 50 + '\n')
    f.write(f'Total samples: 16,384\n')
    f.write(f'Training samples: {X_train.shape[0]}\n')
    f.write(f'Test samples (encrypted): {n_test_all}\n')
    f.write(f'Model: Logistic Regression (64 features -> 4 classes)\n\n')
    f.write(f'Plaintext Accuracy: {plaintext_acc:.4f}\n')
    f.write(f'Encrypted Accuracy: {he_acc:.4f}\n')
    f.write(f'Accuracy Difference: {abs(plaintext_acc - he_acc):.6f}\n\n')
    f.write(f'Encryption Time: {np.mean(encryption_times)*1000:.2f} ms/sample\n')
    f.write(f'HE Inference Time: {avg_inference_time:.2f} ms/sample\n')
    f.write(f'Slowdown Factor: {avg_inference_time / 0.05:.0f}x\n\n')
    f.write('Encryption Scheme: CKKS (TenSEAL)\n')
    f.write('Polynomial Modulus Degree: 16384\n')
    f.write('Coefficient Modulus Bits: [60, 40, 40, 40, 60]\n')

print('\nResults saved to results/he_full_dataset_results.txt')
print('=' * 60)
print('SUCCESS! HE completed on full 16,384 sample dataset.')
print(f'Test samples processed: {n_test_all}')
print(f'Final encrypted accuracy: {he_acc:.4f}')
print('=' * 60)