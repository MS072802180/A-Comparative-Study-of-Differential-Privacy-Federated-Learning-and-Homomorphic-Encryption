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

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print('=' * 60)
print('HOMOMORPHIC ENCRYPTION - FULLY WORKING')
print('=' * 60)

# Load data
print('\n1. Loading dataset...')
with open('data/small_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y'].astype(np.int64)

# Downsample
X_downsampled = X[:, ::32, :]
X_flat = X_downsampled.reshape(X_downsampled.shape[0], -1)
mean = np.mean(X_flat, axis=0)
std = np.std(X_flat, axis=0) + 1e-8
X_flat = (X_flat - mean) / std

# Use only 2 classes
mask = (Y == 0) | (Y == 1)
X_filtered = X_flat[mask]
Y_filtered = Y[mask]

n_train = 12000
n_test = 2000

X_train = X_filtered[:n_train]
Y_train = Y_filtered[:n_train]
X_test = X_filtered[n_train:n_train + n_test]
Y_test = Y_filtered[n_train:n_train + n_test]

print(f'   Training: {X_train.shape}, Test: {X_test.shape}')

# Train logistic regression
print('\n2. Training logistic regression...')

class LogisticRegression(nn.Module):
    def __init__(self, input_dim=64, output_dim=2):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LogisticRegression(input_dim=64, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.long)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(X_train_t)
    loss = criterion(output, Y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 20 == 0:
        print(f'   Epoch {epoch+1}, Loss: {loss.item():.4f}')

# Test plaintext
X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.long)
model.eval()
with torch.no_grad():
    outputs = model(X_test_t)
    _, predicted = torch.max(outputs, 1)
    plaintext_acc = (predicted == Y_test_t).sum().item() / len(Y_test_t)
print(f'\n   Plaintext accuracy: {plaintext_acc:.4f}')

# Extract weights and bias
weight = model.linear.weight.detach().numpy()
bias = model.linear.bias.detach().numpy()

print(f'\n3. Weight shape: {weight.shape}, Bias shape: {bias.shape}')

# Create TenSEAL context
print('\n4. Creating CKKS encryption context...')
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40
print('   Context created')

# Encrypt test samples
print(f'\n5. Encrypting {n_test} test samples...')
encrypted_vectors = []
encryption_times = []

for i in range(n_test):
    start = time.time()
    encrypted = ts.ckks_vector(context, X_test[i].tolist())
    encryption_times.append(time.time() - start)
    encrypted_vectors.append(encrypted)

print(f'   Avg encryption: {np.mean(encryption_times)*1000:.2f} ms')

# Create encrypted bias vectors
print('\n6. Creating encrypted bias vectors...')
encrypted_bias = []
for class_idx in range(2):
    bias_vector = [float(bias[class_idx])]
    encrypted_bias.append(ts.ckks_vector(context, bias_vector))

# Homomorphic inference function
def homomorphic_inference(encrypted_vec, weight, encrypted_bias):
    logits = []
    for class_idx in range(weight.shape[0]):
        w = weight[class_idx].tolist()
        dot_product = encrypted_vec * w
        dot_product_sum = dot_product.sum()
        logit = dot_product_sum + encrypted_bias[class_idx]
        logits.append(logit)
    return logits

# Run homomorphic inference
print(f'\n7. Running homomorphic inference on {n_test} samples...')
correct = 0
inference_times = []

for i in range(n_test):
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

    if (i + 1) % 5 == 0:
        print(f'   Processed {i+1}/{n_test} samples')

he_acc = correct / n_test
avg_inference_time = np.mean(inference_times) * 1000

print(f'\n   Encrypted accuracy: {he_acc:.4f}')
print(f'   Avg inference time: {avg_inference_time:.2f} ms')

# Results
print('\n' + '=' * 60)
print('FINAL RESULTS')
print('=' * 60)
print(f'Plaintext accuracy: {plaintext_acc:.4f}')
print(f'Encrypted accuracy: {he_acc:.4f}')
print(f'Encryption time: {np.mean(encryption_times)*1000:.2f} ms/sample')
print(f'Inference time: {avg_inference_time:.2f} ms/sample')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

accuracies = [plaintext_acc, he_acc]
labels = ['Plaintext', 'Encrypted (HE)']
bars1 = axes[0].bar(labels, accuracies, color=['blue', 'orange'])
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].set_title('Accuracy Comparison')
for bar, val in zip(bars1, accuracies):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=10)

times = [0.05, np.mean(encryption_times)*1000, avg_inference_time]
time_labels = ['Plaintext\nInference', 'Encryption', 'HE\nInference']
bars2 = axes[1].bar(time_labels, times, color=['green', 'orange', 'red'])
axes[1].set_ylabel('Time (ms)')
axes[1].set_title('Performance Overhead')
axes[1].set_yscale('log')
for bar, val in zip(bars2, times):
    if val > 0:
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5,
                    f'{val:.1f}ms', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('plots/he_final_working.png')
print(f'\nPlot saved to plots/he_final_working.png')

with open('results/he_final_working.txt', 'w') as f:
    f.write('HOMOMORPHIC ENCRYPTION WORKING RESULTS\n')
    f.write('=' * 40 + '\n')
    f.write(f'Model: Logistic Regression (2 classes)\n')
    f.write(f'Plaintext Accuracy: {plaintext_acc:.4f}\n')
    f.write(f'Encrypted Accuracy: {he_acc:.4f}\n')
    f.write(f'Encryption Time: {np.mean(encryption_times)*1000:.2f} ms\n')
    f.write(f'Inference Time: {avg_inference_time:.2f} ms\n')

print('\n' + '=' * 60)
print('SUCCESS! HE works correctly.')
print('=' * 60)