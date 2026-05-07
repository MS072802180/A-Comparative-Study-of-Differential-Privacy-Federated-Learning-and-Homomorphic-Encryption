import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print('=' * 60)
print('FINAL RESULTS: ALL THREE TECHNIQUES')
print('=' * 60)

# ============================================================
# DATA FROM YOUR EXPERIMENTS
# ============================================================

# Baseline
baseline_acc = 0.9112

# DIFFERENTIAL PRIVACY
dp_epsilon = [14.69, 2.12, 0.72, 0.25, 0.12]
dp_accuracy = [0.7392, 0.7596, 0.5891, 0.4024, 0.3670]
dp_noise = [0.5, 1.0, 2.0, 5.0, 10.0]

# FEDERATED LEARNING
fl_clients = [2, 5, 10, 20]
fl_accuracy = [0.9200, 0.9161, 0.9039, 0.9069]

# HOMOMORPHIC ENCRYPTION
he_model_names = ['Logistic Regression\n(Linear)', 'MLP with x^2\n(Nonlinear)']
he_model_params = [256, 4352]  # number of parameters as x-axis
he_plain_acc = [0.2850, 0.6900]
he_encrypted_acc = [0.2850, 0.6900]
he_inference_time = [75.83, 1516.13]  # ms
he_encrypt_time = 5.49  # ms per sample

print('\n1. BASELINE')
print(f'   CNN Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)')

print('\n2. DIFFERENTIAL PRIVACY')
for i, eps in enumerate(dp_epsilon):
    print(f'   epsilon = {eps:.2f} -> Accuracy: {dp_accuracy[i]:.4f}')

print('\n3. FEDERATED LEARNING')
for i, clients in enumerate(fl_clients):
    print(f'   {clients} clients -> Accuracy: {fl_accuracy[i]:.4f}')

print('\n4. HOMOMORPHIC ENCRYPTION')
print(f'   Linear Model: Plain={he_plain_acc[0]:.4f}, Encrypted={he_encrypted_acc[0]:.4f}, Params={he_model_params[0]}, Time={he_inference_time[0]:.2f}ms')
print(f'   Nonlinear Model: Plain={he_plain_acc[1]:.4f}, Encrypted={he_encrypted_acc[1]:.4f}, Params={he_model_params[1]}, Time={he_inference_time[1]:.2f}ms')

# ============================================================
# CREATE UNIFIED PLOT - 2x2 GRID
# ============================================================
print('\n5. Generating unified plot (2x2 grid)...')

fig = plt.figure(figsize=(14, 12))
fig.suptitle('Privacy-Preserving Modulation Classification: Results Summary', fontsize=16, fontweight='bold')

# Plot 1: Differential Privacy (line plot)
ax1 = plt.subplot(2, 2, 1)
ax1.plot(dp_epsilon, dp_accuracy, 'bo-', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax1.set_xlabel('Privacy Budget (epsilon) - Smaller = More Private', fontsize=11)
ax1.set_ylabel('Test Accuracy', fontsize=11)
ax1.set_title('(a) Differential Privacy: Accuracy vs Privacy Budget', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=1.5, label=f'Baseline CNN ({baseline_acc:.3f})')
ax1.set_xscale('log')
ax1.legend(loc='lower right')
for i, (x, y) in enumerate(zip(dp_epsilon, dp_accuracy)):
    ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)

# Plot 2: Federated Learning (line plot)
ax2 = plt.subplot(2, 2, 2)
ax2.plot(fl_clients, fl_accuracy, 'gs-', linewidth=2, markersize=8, markerfacecolor='white', markeredgewidth=2)
ax2.set_xlabel('Number of Clients', fontsize=11)
ax2.set_ylabel('Test Accuracy', fontsize=11)
ax2.set_title('(b) Federated Learning: Accuracy vs Number of Clients', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=1.5, label=f'Centralized ({baseline_acc:.3f})')
ax2.legend(loc='lower left')
for i, (x, y) in enumerate(zip(fl_clients, fl_accuracy)):
    ax2.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8)

# Plot 3: Homomorphic Encryption - Accuracy vs Model Complexity (line plot)
ax3 = plt.subplot(2, 2, 3)
ax3.plot(he_model_params, he_encrypted_acc, 'mo-', linewidth=2, markersize=10, markerfacecolor='white', markeredgewidth=2)
ax3.set_xlabel('Model Complexity (Number of Parameters)', fontsize=11)
ax3.set_ylabel('Test Accuracy', fontsize=11)
ax3.set_title('(c) Homomorphic Encryption: Accuracy vs Model Complexity', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log')
for i, (x, y) in enumerate(zip(he_model_params, he_encrypted_acc)):
    ax3.annotate(f'{he_model_names[i].replace(chr(10), " ")}\n({y:.3f})', 
                 (x, y), textcoords="offset points", xytext=(5,10), ha='center', fontsize=8)

# Also mark the plaintext baseline for each model
for i, (x, y) in enumerate(zip(he_model_params, he_plain_acc)):
    ax3.plot(x, y, 'co', markersize=6)
ax3.annotate('Plaintext\nLinear', (256, 0.285), textcoords="offset points", xytext=(-30, -15), fontsize=8)
ax3.annotate('Plaintext\nNonlinear', (4352, 0.69), textcoords="offset points", xytext=(10, -15), fontsize=8)

# Plot 4: Homomorphic Encryption - Accuracy vs Inference Time (scatter plot)
ax4 = plt.subplot(2, 2, 4)
colors_he = ['orange', 'red']
sizes = [100, 200]
for i in range(len(he_model_names)):
    ax4.scatter(he_inference_time[i], he_encrypted_acc[i], 
                s=sizes[i], c=colors_he[i], marker='o', edgecolor='black', linewidth=1.5, zorder=5)
    ax4.annotate(he_model_names[i].replace(chr(10), ' '), 
                 (he_inference_time[i], he_encrypted_acc[i]), 
                 textcoords="offset points", xytext=(10, 5), ha='left', fontsize=9)

ax4.set_xlabel('HE Inference Time (ms) - Log Scale', fontsize=11)
ax4.set_ylabel('Encrypted Accuracy', fontsize=11)
ax4.set_title('(d) HE: Accuracy vs Inference Time Trade-off', fontsize=12)
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)
ax4.set_ylim([0, 1])

# Add a dashed line showing plaintext baseline
ax4.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=1.5, alpha=0.5, label=f'CNN Baseline ({baseline_acc:.3f})')
ax4.legend(loc='upper left', fontsize=9)

plt.tight_layout()
plt.savefig('plots/final_all_techniques.png', dpi=150, bbox_inches='tight')
print('   Plot saved to plots/final_all_techniques.png')

# ============================================================
# ALSO CREATE A DEDICATED HE PLOT (like DP and FL)
# ============================================================
print('\n6. Creating dedicated HE plot (line plot like DP and FL)...')

fig_he, axes_he = plt.subplots(1, 2, figsize=(12, 5))
fig_he.suptitle('Homomorphic Encryption: Detailed Results', fontsize=14, fontweight='bold')

# Left: Accuracy vs Model Parameters
ax_he1 = axes_he[0]
ax_he1.plot(he_model_params, he_plain_acc, 'b--o', linewidth=2, markersize=8, label='Plaintext Accuracy', markerfacecolor='white')
ax_he1.plot(he_model_params, he_encrypted_acc, 'r-o', linewidth=2, markersize=8, label='Encrypted Accuracy', markerfacecolor='white')
ax_he1.set_xlabel('Model Complexity (Number of Parameters)', fontsize=11)
ax_he1.set_ylabel('Accuracy', fontsize=11)
ax_he1.set_title('Accuracy vs Model Complexity', fontsize=12)
ax_he1.set_xscale('log')
ax_he1.grid(True, alpha=0.3)
ax_he1.legend()
for i, x in enumerate(he_model_params):
    ax_he1.annotate(f'{he_model_names[i].replace(chr(10), " ")}\nPlain:{he_plain_acc[i]:.3f}', 
                    (x, he_plain_acc[i]), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)
    ax_he1.annotate(f'Enc:{he_encrypted_acc[i]:.3f}', 
                    (x, he_encrypted_acc[i]), textcoords="offset points", xytext=(5, -12), ha='left', fontsize=8)

# Right: Inference Time vs Model Parameters
ax_he2 = axes_he[1]
ax_he2.bar(he_model_names, he_inference_time, color=['orange', 'red'], alpha=0.7, edgecolor='black')
ax_he2.set_xlabel('Model Type', fontsize=11)
ax_he2.set_ylabel('Inference Time (ms) - Log Scale', fontsize=11)
ax_he2.set_title('Inference Time vs Model Complexity', fontsize=12)
ax_he2.set_yscale('log')
ax_he2.grid(True, alpha=0.3, axis='y')
for i, (name, val) in enumerate(zip(he_model_names, he_inference_time)):
    ax_he2.annotate(f'{val:.1f} ms', (i, val), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('plots/he_detailed_results.png', dpi=150, bbox_inches='tight')
print('   Dedicated HE plot saved to plots/he_detailed_results.png')

# ============================================================
# SAVE RESULTS TABLE (NO UNICODE)
# ============================================================
print('\n7. Saving comparison table...')

with open('results/final_all_results.txt', 'w') as f:
    f.write('=' * 70 + '\n')
    f.write('FINAL RESULTS: PRIVACY-PRESERVING MODULATION CLASSIFICATION\n')
    f.write('=' * 70 + '\n\n')
    
    f.write('BASELINE MODEL:\n')
    f.write(f'  Architecture: CNN (3 conv layers, BatchNorm, Dropout)\n')
    f.write(f'  Dataset: 16,384 samples, 4 modulation classes, SNR = 10 dB\n')
    f.write(f'  Test Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)\n\n')
    
    f.write('=' * 70 + '\n')
    f.write('DIFFERENTIAL PRIVACY (DP-SGD with Opacus)\n')
    f.write('=' * 70 + '\n')
    f.write('  Privacy Budget (epsilon) -> Test Accuracy:\n')
    for i, eps in enumerate(dp_epsilon):
        f.write(f'    epsilon = {eps:.2f} (noise={dp_noise[i]}): {dp_accuracy[i]:.4f}\n')
    f.write('  Key Insight: Stronger privacy (smaller epsilon) reduces accuracy\n\n')
    
    f.write('=' * 70 + '\n')
    f.write('FEDERATED LEARNING (FedAvg with Flower)\n')
    f.write('=' * 70 + '\n')
    f.write('  Number of Clients -> Test Accuracy:\n')
    for i, clients in enumerate(fl_clients):
        f.write(f'    {clients} clients: {fl_accuracy[i]:.4f}\n')
    f.write('  Key Insight: FL preserves accuracy well (within 1 percent of centralized)\n\n')
    
    f.write('=' * 70 + '\n')
    f.write('HOMOMORPHIC ENCRYPTION (CKKS with TenSEAL)\n')
    f.write('=' * 70 + '\n')
    f.write('  Encryption Scheme: CKKS\n')
    f.write('  Polynomial Modulus Degree: 16384\n')
    f.write(f'  Encryption Time: {he_encrypt_time:.2f} ms per sample\n\n')
    
    f.write('  Model 1: Logistic Regression (Linear)\n')
    f.write(f'    Parameters: {he_model_params[0]}\n')
    f.write(f'    Plaintext Accuracy: {he_plain_acc[0]:.4f}\n')
    f.write(f'    Encrypted Accuracy: {he_encrypted_acc[0]:.4f}\n')
    f.write(f'    HE Inference Time: {he_inference_time[0]:.2f} ms\n')
    f.write(f'    Slowdown: {he_inference_time[0]/0.05:.0f}x\n\n')
    
    f.write('  Model 2: MLP with x^2 Activation (Non-linear)\n')
    f.write(f'    Architecture: 64 -> 64 -> 4\n')
    f.write(f'    Activation: x^2 (polynomial, HE-compatible)\n')
    f.write(f'    Parameters: {he_model_params[1]}\n')
    f.write(f'    Plaintext Accuracy: {he_plain_acc[1]:.4f}\n')
    f.write(f'    Encrypted Accuracy: {he_encrypted_acc[1]:.4f}\n')
    f.write(f'    HE Inference Time: {he_inference_time[1]:.2f} ms\n')
    f.write(f'    Slowdown: {he_inference_time[1]/0.05:.0f}x\n\n')
    
    f.write('=' * 70 + '\n')
    f.write('CONCLUSIONS\n')
    f.write('=' * 70 + '\n')
    f.write('1. Differential Privacy: Provides formal privacy guarantees but reduces accuracy.\n')
    f.write('   Best for: Centralized training where regulatory compliance is required.\n\n')
    f.write('2. Federated Learning: Preserves accuracy well but requires distributed infrastructure.\n')
    f.write('   Best for: Multi-operator collaboration where data cannot be centralized.\n\n')
    f.write('3. Homomorphic Encryption: Strongest input privacy but highest computational cost.\n')
    f.write('   Best for: User-side privacy where inputs must remain encrypted in transit.\n\n')
    f.write('4. Non-linear HE (x^2 activation) achieves 69.0% accuracy vs 28.5% for linear.\n')
    f.write('   However, inference time is 20x slower (1516ms vs 76ms).\n\n')
    f.write('5. No single technique dominates. Choice depends on:\n')
    f.write('   - Required privacy level\n')
    f.write('   - Available computational resources\n')
    f.write('   - Deployment architecture (centralized vs distributed)\n')
    f.write('   - Real-time constraints\n')

print('   Table saved to results/final_all_results.txt')

# ============================================================
# CREATE COMPARISON BAR CHART FOR ALL TECHNIQUES
# ============================================================
print('\n8. Creating technique comparison bar chart...')

fig2, ax = plt.subplots(figsize=(12, 6))

techniques = ['Baseline\nCNN', 'DP\n(eps=2.12)', 'FL\n(10 clients)', 'HE Linear\n(Logistic)', 'HE Nonlinear\n(MLP)']
acc_values = [baseline_acc, dp_accuracy[1], fl_accuracy[2], he_encrypted_acc[0], he_encrypted_acc[1]]
colors = ['green', 'orange', 'blue', 'red', 'purple']

bars = ax.bar(techniques, acc_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_xlabel('Technique', fontsize=12)
ax.set_title('Comparison of Privacy-Preserving Techniques for Modulation Classification', fontsize=14)
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, acc_values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/technique_comparison_bar.png', dpi=150)
print('   Bar chart saved to plots/technique_comparison_bar.png')

print('\n' + '=' * 60)
print('FINAL RESULTS COMPLETE')
print('=' * 60)
print('\nFiles generated:')
print('  - plots/final_all_techniques.png (2x2 subplot figure)')
print('  - plots/he_detailed_results.png (dedicated HE plot like DP and FL)')
print('  - plots/technique_comparison_bar.png (bar chart)')
print('  - results/final_all_results.txt (complete results table)')
print('=' * 60)