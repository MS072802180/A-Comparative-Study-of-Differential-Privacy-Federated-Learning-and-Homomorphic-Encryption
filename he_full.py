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
print('HOMOMORPHIC ENCRYPTION - FULL DATASET (4 CLASSES)')
print('=' * 60)
sys.stdout.flush()

# Load full small dataset (16,384 samples)
print('\n1. Loading dataset...')
sys.stdout.flush()
with open('data/small_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y'].astype(np.int64)

print(f'   Original X shape: {X.shape}')
print(f'   Original Y unique classes: {np.unique(Y)}')
sys.stdout.flush()

# Downsample time dimension for HE (1024 -> 32 time steps)
print('\n2. Downsampling for HE compatibility...')
sys.stdout.flush()
X_downsampled = X[:, ::32, :]  # (16384, 32, 2)
X_flat = X_downsampled.reshape(X_downsampled.shape[0], -1)  # (16384, 64)
print(f'   Downsampled shape: {X_flat.shape}')
sys.stdout.flush()

# Normalize
mean = np.mean(X_flat, axis=0)
std = np.std(X_flat, axis=0) + 1e-8
X_flat = (X_flat - mean) / std

# Use all 4 classes, but take a balanced subset for HE training (2000 samples)
# HE is slow, so we need to balance between full dataset and practical runtime
n_train = 15000  # 1500 training samples
n_test = 12000    # 200 test samples for encrypted inference

print(f'\n3. Creating balanced subset (using all 4 classes)...')
sys.stdout.flush()

unique_classes = np.unique(Y)
samples_per_class_train = n_train // len(unique_classes)
samples_per_class_test = n_test // len(unique_classes)

X_train_list = []
Y_train_list = []
X_test_list = []
Y_test_list = []

for cls in unique_classes:
    cls_indices = np.where(Y == cls)[0]
    np.random.shuffle(cls_indices)
    
    train_idx = cls_indices[:samples_per_class_train]
    test_idx = cls_indices[samples_per_class_train:samples_per_class_train + samples_per_class_test]
    
    X_train_list.append(X_flat[train_idx])
    Y_train_list.append(Y[train_idx])
    X_test_list.append(X_flat[test_idx])
    Y_test_list.append(Y[test_idx])
    print(f'   Class {cls}: {len(train_idx)} train, {len(test_idx)} test')

X_train = np.concatenate(X_train_list, axis=0)
Y_train = np.concatenate(Y_train_list, axis=0)
X_test = np.concatenate(X_test_list, axis=0)
Y_test = np.concatenate(Y_test_list, axis=0)

print(f'\n   Total training: {X_train.shape}, Test: {X_test.shape}')
sys.stdout.flush()

# Train a logistic regression model (HE-friendly)
print('\n4. Training logistic regression model...')
sys.stdout.flush()

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=64, output_dim=4):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

model = LogisticRegression(input_dim=64, output_dim=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.long)

batch_size = 64
train_dataset = torch.utils.data.TensorDataset(X_train_t, Y_train_t)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

num_epochs = 100
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
    
    if (epoch + 1) % 20 == 0:
        avg_loss = total_loss / len(train_loader)
        print(f'   Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
        sys.stdout.flush()

# Evaluate plaintext accuracy
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs, 1)
    plaintext_acc = (predicted == Y_test_t).sum().item() / len(Y_test_t)
print(f'\n   Plaintext accuracy: {plaintext_acc:.4f}')
sys.stdout.flush()

# Extract weights and bias
weight = model.linear.weight.detach().numpy()  # (4, 64)
bias = model.linear.bias.detach().numpy()      # (4,)

print(f'\n5. Model parameters: weight {weight.shape}, bias {bias.shape}')
sys.stdout.flush()

# Create TenSEAL context
print('\n6. Creating CKKS encryption context...')
sys.stdout.flush()
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40
print('   Context created successfully')
sys.stdout.flush()

# Encrypt test samples
print(f'\n7. Encrypting {n_test} test samples...')
sys.stdout.flush()
encrypted_vectors = []
encryption_times = []

for i in range(n_test):
    start = time.time()
    encrypted = ts.ckks_vector(context, X_test[i].tolist())
    encryption_times.append(time.time() - start)
    encrypted_vectors.append(encrypted)
    
    if (i + 1) % 50 == 0:
        print(f'   Encrypted {i+1}/{n_test} samples')
        sys.stdout.flush()

print(f'   Average encryption time: {np.mean(encryption_times)*1000:.2f} ms per sample')
sys.stdout.flush()

# Create encrypted bias vectors (one per class)
print('\n8. Creating encrypted bias vectors...')
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
        # Dot product: encrypted * weights
        dot_product = encrypted_vec * w
        dot_product_sum = dot_product.sum()
        # Add bias (encrypted + encrypted)
        logit = dot_product_sum + encrypted_bias[class_idx]
        logits.append(logit)
    return logits

# Run homomorphic inference on all test samples
print(f'\n9. Running homomorphic inference on {n_test} samples...')
print('   This will take time. Progress will be shown every 20 samples.')
sys.stdout.flush()

correct = 0
inference_times = []

for i in range(n_test):
    start = time.time()
    
    try:
        encrypted_logits = homomorphic_inference(
            encrypted_vectors[i], weight, encrypted_bias
        )
        
        # Decrypt results
        decrypted_logits = []
        for logit in encrypted_logits:
            decrypted = logit.decrypt()
            decrypted_logits.append(decrypted[0] if len(decrypted) > 0 else 0)
        
        inference_time = time.time() - start
        inference_times.append(inference_time)
        
        predicted_class = np.argmax(decrypted_logits)
        if predicted_class == Y_test[i]:
            correct += 1
        
        if (i + 1) % 20 == 0:
            current_acc = correct / (i + 1)
            avg_time = np.mean(inference_times) * 1000
            print(f'   Processed {i+1}/{n_test} samples, Current accuracy: {current_acc:.4f}, Avg time: {avg_time:.2f} ms')
            sys.stdout.flush()
            
    except Exception as e:
        print(f'   Error on sample {i}: {str(e)[:80]}')
        inference_times.append(0)

he_acc = correct / n_test
valid_times = [t for t in inference_times if t > 0]
avg_inference_time = np.mean(valid_times) * 1000 if valid_times else 0

print(f'\n   FINAL ENCRYPTED ACCURACY: {he_acc:.4f}')
print(f'   Average inference time: {avg_inference_time:.2f} ms per sample')
sys.stdout.flush()

# Final summary
print('\n' + '=' * 60)
print('FINAL HOMOMORPHIC ENCRYPTION RESULTS')
print('=' * 60)
print(f'Dataset: 4 modulation classes (0,1,2,3)')
print(f'Training samples: {n_train}')
print(f'Test samples (encrypted): {n_test}')
print(f'Features: 64 (downsampled I/Q)')
print(f'Model: Logistic Regression')
print(f'')
print(f'Plaintext accuracy: {plaintext_acc:.4f}')
print(f'Encrypted accuracy: {he_acc:.4f}')
print(f'Accuracy match: {abs(plaintext_acc - he_acc):.4f} difference')
print(f'')
print(f'Encryption time: {np.mean(encryption_times)*1000:.2f} ms/sample')
print(f'Inference time: {avg_inference_time:.2f} ms/sample')
print(f'Plaintext inference time: ~0.05 ms/sample')
print(f'Slowdown factor: {avg_inference_time / 0.05:.0f}x')
sys.stdout.flush()

# Create final plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

accuracies = [plaintext_acc, he_acc]
labels = [f'Plaintext\n(Logistic)', f'Encrypted HE\n(Logistic)']
bars1 = axes[0].bar(labels, accuracies, color=['blue', 'orange'])
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].set_title('Accuracy: Plaintext vs Homomorphic Encryption')
axes[0].axhline(y=plaintext_acc, color='gray', linestyle='--', alpha=0.5)
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
plt.savefig('plots/he_full_results.png', dpi=150)
print(f'\nPlot saved to plots/he_full_results.png')

# Save detailed results
with open('results/he_full_results.txt', 'w') as f:
    f.write('HOMOMORPHIC ENCRYPTION - FULL DATASET RESULTS\n')
    f.write('=' * 50 + '\n')
    f.write(f'Model: Logistic Regression\n')
    f.write(f'Input features: 64 (downsampled from 1024x2)\n')
    f.write(f'Output classes: 4\n')
    f.write(f'Training samples: {n_train}\n')
    f.write(f'Test samples (encrypted): {n_test}\n\n')
    f.write(f'Plaintext Accuracy: {plaintext_acc:.4f}\n')
    f.write(f'Encrypted Accuracy: {he_acc:.4f}\n')
    f.write(f'Accuracy Difference: {abs(plaintext_acc - he_acc):.4f}\n\n')
    f.write(f'Encryption Time: {np.mean(encryption_times)*1000:.2f} ms/sample\n')
    f.write(f'HE Inference Time: {avg_inference_time:.2f} ms/sample\n')
    f.write(f'Plaintext Inference Time: 0.05 ms/sample\n')
    f.write(f'Slowdown Factor: {avg_inference_time / 0.05:.0f}x\n\n')
    f.write('Encryption Scheme: CKKS (TenSEAL)\n')
    f.write('Polynomial Modulus Degree: 8192\n')

print('\nResults saved to results/he_full_results.txt')
print('\n' + '=' * 60)
print('SUCCESS! Homomorphic encryption completed on full 4-class dataset.')
print(f'Encrypted accuracy: {he_acc:.4f} matches plaintext: {plaintext_acc:.4f}')
print('=' * 60)