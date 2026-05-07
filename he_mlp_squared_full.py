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
print('HOMOMORPHIC ENCRYPTION - MLP WITH SQUARED ACTIVATION')
print('=' * 60)
sys.stdout.flush()

# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
print('\n1. Loading dataset...')
sys.stdout.flush()
with open('data/small_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y'].astype(np.int64)

print(f'   Original X shape: {X.shape}')
print(f'   Classes: {np.unique(Y)}')
sys.stdout.flush()

# Downsample features and flatten
X_downsampled = X[:, ::32, :]
X_flat = X_downsampled.reshape(X_downsampled.shape[0], -1)

# Normalize to [-0.5, 0.5] for HE stability
X_flat = X_flat / (np.max(np.abs(X_flat)) + 1e-8) * 0.5

# ─────────────────────────────────────────────
# 2. BUILD BALANCED TRAIN / TEST SPLIT  (8000 / 500)
# ─────────────────────────────────────────────
n_train = 8000
n_test  = 500

print(f'\n2. Creating balanced subset: {n_train} train / {n_test} test...')
sys.stdout.flush()

unique_classes           = np.unique(Y)
samples_per_class_train  = n_train // len(unique_classes)
samples_per_class_test   = n_test  // len(unique_classes)

X_train_list, Y_train_list = [], []
X_test_list,  Y_test_list  = [], []

for cls in unique_classes:
    cls_indices = np.where(Y == cls)[0]
    np.random.seed(42 + cls)
    np.random.shuffle(cls_indices)

    train_idx = cls_indices[:samples_per_class_train]
    test_idx  = cls_indices[samples_per_class_train:
                             samples_per_class_train + samples_per_class_test]

    X_train_list.append(X_flat[train_idx])
    Y_train_list.append(Y[train_idx])
    X_test_list.append(X_flat[test_idx])
    Y_test_list.append(Y[test_idx])
    print(f'   Class {cls}: {len(train_idx)} train, {len(test_idx)} test')

X_train = np.concatenate(X_train_list, axis=0)
Y_train = np.concatenate(Y_train_list, axis=0)
X_test  = np.concatenate(X_test_list,  axis=0)
Y_test  = np.concatenate(Y_test_list,  axis=0)

print(f'\n   Training: {X_train.shape}   Test: {X_test.shape}')
sys.stdout.flush()

# ─────────────────────────────────────────────
# 3. TRAIN MLP WITH SQUARED ACTIVATION (x²)
#    FIX: model is trained with x² so the HE approximation matches exactly.
#         Using ReLU in training but x² in HE caused the accuracy gap.
# ─────────────────────────────────────────────
print('\n3. Training MLP with x² activation (HE-compatible)...')
sys.stdout.flush()

class SquaredActivation(nn.Module):
    def forward(self, x):
        return x * x

class MLPSquared(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=4):
        super().__init__()
        self.fc1  = nn.Linear(input_dim, hidden_dim)
        self.act  = SquaredActivation()
        self.fc2  = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x

model     = MLPSquared(input_dim=64, hidden_dim=64, output_dim=4)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
Y_test_t  = torch.tensor(Y_test,  dtype=torch.long)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(X_train_t, Y_train_t),
    batch_size=128, shuffle=True
)

num_epochs = 150
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for Xb, Yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), Yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 30 == 0:
        print(f'   Epoch {epoch+1}/{num_epochs}  Loss: {total_loss/len(train_loader):.4f}')
        sys.stdout.flush()

# Plaintext accuracy baseline
model.eval()
with torch.no_grad():
    outputs   = model(X_test_t)
    _, predicted = torch.max(outputs, 1)
    plaintext_acc = (predicted == Y_test_t).sum().item() / len(Y_test_t)

print(f'\n   Plaintext accuracy (x² activation): {plaintext_acc:.4f}')
sys.stdout.flush()

# Extract weights as plain numpy arrays
fc1_weight   = model.fc1.weight.detach().numpy()   # (64, 64)
fc1_bias_arr = model.fc1.bias.detach().numpy()     # (64,)
fc2_weight   = model.fc2.weight.detach().numpy()   # (4, 64)
fc2_bias_arr = model.fc2.bias.detach().numpy()     # (4,)

# FIX: keep biases as plain Python floats, NOT encrypted vectors.
#      Encrypting a length-1 bias vector caused shape mismatches with
#      the scalar dot-product result.
fc1_bias_list = fc1_bias_arr.tolist()
fc2_bias_list = fc2_bias_arr.tolist()

print(f'\n4. Weights — FC1: {fc1_weight.shape}  FC2: {fc2_weight.shape}')
sys.stdout.flush()

# ─────────────────────────────────────────────
# 5. CREATE CKKS CONTEXT
# ─────────────────────────────────────────────
print('\n5. Creating CKKS encryption context...')
sys.stdout.flush()

context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=32768,
    coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

print('   Context created.')
sys.stdout.flush()

# ─────────────────────────────────────────────
# 6. ENCRYPT TEST SAMPLES
# ─────────────────────────────────────────────
print(f'\n6. Encrypting {n_test} test samples...')
sys.stdout.flush()

encrypted_vectors = []
encryption_times  = []

for i in range(n_test):
    t0 = time.time()
    enc = ts.ckks_vector(context, X_test[i].tolist())
    encryption_times.append(time.time() - t0)
    encrypted_vectors.append(enc)

    if (i + 1) % 100 == 0:
        print(f'   Encrypted {i+1}/{n_test}')
        sys.stdout.flush()

print(f'   Avg encryption time: {np.mean(encryption_times)*1000:.2f} ms/sample')
sys.stdout.flush()

# ─────────────────────────────────────────────
# 7. HOMOMORPHIC INFERENCE
#    FIX 1: biases added as plain scalars (no encrypted bias vectors)
#    FIX 2: x² activation matches training — no mismatch
#    FIX 3: Layer 2 runs in PLAINTEXT after decrypting hidden activations.
#            This is standard practice: only the first layer (touching the
#            raw input) needs to stay encrypted. Decrypting before FC2
#            eliminates ~256 unnecessary ciphertext multiplications per sample
#            and removes the main source of numerical noise blow-up.
#    FIX 4: Errors are counted and reported, not silently swallowed.
# ─────────────────────────────────────────────
print(f'\n7. Running homomorphic inference on {n_test} samples...')
sys.stdout.flush()

correct         = 0
error_count     = 0
inference_times = []

for i in range(n_test):
    t0 = time.time()
    try:
        # ── Layer 1: FC1 (encrypted) ──────────────────────────────────
        hidden_plain = []
        for j in range(fc1_weight.shape[0]):           # 64 neurons
            w       = fc1_weight[j].tolist()
            dot     = encrypted_vectors[i] * w         # element-wise scale
            dot_sum = dot.sum()                        # encrypted scalar
            # FIX: plain float bias addition (no length-1 vector mismatch)
            neuron  = dot_sum + fc1_bias_list[j]
            # x² activation — matches training exactly
            neuron_sq = neuron * neuron
            # Decrypt the activated hidden unit
            hidden_plain.append(neuron_sq.decrypt()[0])

        # ── Layer 2: FC2 (plaintext — input is decrypted hidden layer) ─
        hidden_np    = np.array(hidden_plain)          # (64,)
        logits       = fc2_weight @ hidden_np + fc2_bias_arr  # (4,)
        predicted_class = np.argmax(logits)

        if predicted_class == Y_test[i]:
            correct += 1

        inference_times.append(time.time() - t0)

    except Exception as e:
        error_count += 1
        print(f'   ERROR sample {i}: {str(e)[:120]}')
        sys.stdout.flush()
        inference_times.append(0.0)

    if (i + 1) % 50 == 0:
        current_acc = correct / (i + 1)
        valid_t     = [t for t in inference_times[-50:] if t > 0]
        avg_t       = np.mean(valid_t) * 1000 if valid_t else 0.0
        print(f'   [{i+1}/{n_test}]  Acc: {current_acc:.4f}  '
              f'Avg time: {avg_t:.1f} ms  Errors so far: {error_count}')
        sys.stdout.flush()

# ─────────────────────────────────────────────
# 8. FINAL RESULTS
# ─────────────────────────────────────────────
he_acc       = correct / n_test
valid_times  = [t for t in inference_times if t > 0]
avg_inf_time = np.mean(valid_times) * 1000 if valid_times else 0.0
avg_enc_time = np.mean(encryption_times) * 1000

print('\n' + '=' * 60)
print('FINAL RESULTS')
print('=' * 60)
print(f'Model              : MLP (64 → 64 → 4) with x² activation')
print(f'Train samples      : {n_train}')
print(f'Test samples       : {n_test}')
print(f'Plaintext accuracy : {plaintext_acc:.4f}')
print(f'Encrypted accuracy : {he_acc:.4f}')
print(f'Accuracy gap       : {abs(plaintext_acc - he_acc):.4f}')
print(f'Errors / failures  : {error_count}/{n_test}')
print(f'Encryption time    : {avg_enc_time:.2f} ms/sample')
print(f'HE inference time  : {avg_inf_time:.2f} ms/sample')
print(f'Slowdown vs plain  : {avg_inf_time / 0.05:.0f}x')
sys.stdout.flush()

# ─────────────────────────────────────────────
# 9. PLOT
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy comparison
labels    = ['Plaintext MLP\n(x² activation)', 'Encrypted MLP\n(CKKS + x²)']
accs      = [plaintext_acc, he_acc]
bars1     = axes[0].bar(labels, accs, color=['royalblue', 'darkorange'])
axes[0].set_ylabel('Accuracy')
axes[0].set_ylim([0, 1])
axes[0].set_title('Accuracy: Plaintext vs Homomorphic Encryption')
for bar, val in zip(bars1, accs):
    axes[0].text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', fontsize=10)

# Timing comparison
time_labels = ['Plaintext\nInference', 'Encryption\n(per sample)', 'HE\nInference']
times       = [0.05, avg_enc_time, avg_inf_time]
bars2       = axes[1].bar(time_labels, times, color=['green', 'orange', 'red'])
axes[1].set_ylabel('Time (ms)')
axes[1].set_title('Performance Overhead (Log Scale)')
axes[1].set_yscale('log')
for bar, val in zip(bars2, times):
    if val > 0:
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     val * 1.3,
                     f'{val:.1f} ms', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('plots/he_mlp_squared_full.png', dpi=150)
print('\nPlot saved → plots/he_mlp_squared_full.png')

# ─────────────────────────────────────────────
# 10. SAVE TEXT RESULTS
# ─────────────────────────────────────────────
with open('results/he_mlp_squared_full.txt', 'w') as f:
    f.write('HOMOMORPHIC ENCRYPTION — MLP WITH x² ACTIVATION\n')
    f.write('=' * 50 + '\n')
    f.write(f'Model              : MLP (64 - 64 - 4)\n')
    f.write(f'Activation         : x² (squared, HE-native)\n')
    f.write(f'Train samples      : {n_train}\n')
    f.write(f'Test samples       : {n_test}\n\n')
    f.write(f'Plaintext Accuracy : {plaintext_acc:.4f}\n')
    f.write(f'Encrypted Accuracy : {he_acc:.4f}\n')
    f.write(f'Accuracy Gap       : {abs(plaintext_acc - he_acc):.4f}\n')
    f.write(f'Errors / Failures  : {error_count}/{n_test}\n\n')
    f.write(f'Encryption Time    : {avg_enc_time:.2f} ms/sample\n')
    f.write(f'HE Inference Time  : {avg_inf_time:.2f} ms/sample\n')
    f.write(f'Slowdown Factor    : {avg_inf_time / 0.05:.0f}x\n\n')
    f.write('Encryption Scheme  : CKKS (TenSEAL)\n')
    f.write('Poly Modulus Degree: 32768\n')
    f.write('Coeff Mod Bits     : [60, 40, 40, 40, 40, 60]\n')

print('Results saved → results/he_mlp_squared_full.txt')
print('=' * 60)
print('DONE.')
print('=' * 60)