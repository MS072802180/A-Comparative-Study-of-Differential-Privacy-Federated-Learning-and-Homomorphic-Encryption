import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

print('=' * 60)
print('FINAL RESULTS SUMMARY FOR PAPER')
print('=' * 60)

baseline_acc = 0.9112

dp_epsilons = [14.69, 2.12, 0.72, 0.25, 0.12]
dp_accuracies = [0.7392, 0.7596, 0.5891, 0.4024, 0.3670]
dp_noise = [0.5, 1.0, 2.0, 5.0, 10.0]

fl_clients = [2, 5, 10, 20]
fl_accuracies = [0.9200, 0.9161, 0.9039, 0.9069]

he_encrypt_time_ms = 2.46
he_decrypt_time_ms = 1.00
he_linear_acc = 0.3500

print(f'\n1. BASELINE MODEL')
print(f'   Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)')

print(f'\n2. DIFFERENTIAL PRIVACY')
print(f'   Baseline: {baseline_acc:.4f}')
print(f'   Noise -> Epsilon -> Accuracy:')
for i in range(len(dp_noise)):
    print(f'     Noise={dp_noise[i]}: epsilon={dp_epsilons[i]:.2f}, acc={dp_accuracies[i]:.4f}')

print(f'\n3. FEDERATED LEARNING')
print(f'   Centralized baseline: {baseline_acc:.4f}')
for i in range(len(fl_clients)):
    print(f'     {fl_clients[i]} clients: {fl_accuracies[i]:.4f}')

print(f'\n4. HOMOMORPHIC ENCRYPTION')
print(f'   Linear model accuracy: {he_linear_acc:.4f}')
print(f'   Encryption time: {he_encrypt_time_ms:.2f} ms/sample')
print(f'   Decryption time: {he_decrypt_time_ms:.2f} ms/sample')
print(f'   Note: Full CNN HE would be 100-1000x slower (cited from literature)')

# Create final comparison plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: DP trade-off
axes[0, 0].plot(dp_epsilons, dp_accuracies, 'bo-', linewidth=2, markersize=8)
axes[0, 0].set_xlabel('Privacy Budget (Epsilon - smaller = more private)', fontsize=10)
axes[0, 0].set_ylabel('Test Accuracy', fontsize=10)
axes[0, 0].set_title('Differential Privacy: Privacy vs Accuracy', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axhline(y=baseline_acc, color='r', linestyle='--', label=f'Baseline ({baseline_acc:.3f})')
axes[0, 0].legend()
axes[0, 0].set_xscale('log')

# Plot 2: FL results
axes[0, 1].plot(fl_clients, fl_accuracies, 'gs-', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Number of Clients', fontsize=10)
axes[0, 1].set_ylabel('Test Accuracy', fontsize=10)
axes[0, 1].set_title('Federated Learning: Accuracy vs Number of Clients', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=baseline_acc, color='r', linestyle='--', label=f'Centralized ({baseline_acc:.3f})')
axes[0, 1].legend()

# Plot 3: Summary bar chart of all techniques
techniques = ['Baseline', 'DP\n(ε=2.12)', 'FL\n(10 clients)', 'HE\n(linear)']
acc_values = [baseline_acc, dp_accuracies[1], fl_accuracies[2], he_linear_acc]
colors = ['green', 'orange', 'blue', 'red']
bars = axes[1, 0].bar(techniques, acc_values, color=colors)
axes[1, 0].set_ylabel('Accuracy', fontsize=10)
axes[1, 0].set_title('Comparative Accuracy Summary', fontsize=12)
axes[1, 0].set_ylim([0, 1])
for bar, val in zip(bars, acc_values):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{val:.3f}', ha='center', fontsize=9)

# Plot 4: HE timing
he_times = [0.05, he_encrypt_time_ms, he_decrypt_time_ms]
he_labels = ['Plaintext\nInference', 'Encryption', 'Decryption']
bars_times = axes[1, 1].bar(he_labels, he_times, color=['green', 'orange', 'blue'])
axes[1, 1].set_ylabel('Time (ms)', fontsize=10)
axes[1, 1].set_title('Homomorphic Encryption Overhead', fontsize=12)
axes[1, 1].set_yscale('log')
for bar, val in zip(bars_times, he_times):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{val:.2f}ms', ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('plots/final_comparison.png', dpi=150)
print(f'\nPlot saved to plots/final_comparison.png')

# Save all results to a single file
with open('results/all_results.txt', 'w') as f:
    f.write('=' * 60 + '\n')
    f.write('PRIVACY-PRESERVING MODULATION CLASSIFICATION\n')
    f.write('EXPERIMENTAL RESULTS\n')
    f.write('=' * 60 + '\n\n')
    
    f.write('DATASET:\n')
    f.write('  Source: RadioML 2018.01A\n')
    f.write('  Samples used: 16,384\n')
    f.write('  Modulation classes: 4\n')
    f.write('  SNR: 10 dB\n\n')
    
    f.write('BASELINE MODEL:\n')
    f.write('  Architecture: CNN (3 conv layers, BatchNorm, Dropout)\n')
    f.write(f'  Test accuracy: {baseline_acc:.4f} ({baseline_acc*100:.2f}%)\n\n')
    
    f.write('DIFFERENTIAL PRIVACY:\n')
    f.write('  Using Opacus with GroupNorm (DP-SGD)\n')
    f.write(f'  Baseline: {baseline_acc:.4f}\n')
    f.write('  Noise -> Epsilon -> Accuracy:\n')
    for i in range(len(dp_noise)):
        f.write(f'    Noise={dp_noise[i]}: epsilon={dp_epsilons[i]:.2f}, accuracy={dp_accuracies[i]:.4f}\n')
    f.write('\n')
    
    f.write('FEDERATED LEARNING:\n')
    f.write('  Using FedAvg with 10 rounds, 3 local epochs\n')
    f.write(f'  Centralized baseline: {baseline_acc:.4f}\n')
    f.write('  Clients -> Accuracy:\n')
    for i in range(len(fl_clients)):
        f.write(f'    {fl_clients[i]} clients: {fl_accuracies[i]:.4f}\n')
    f.write('\n')
    
    f.write('HOMOMORPHIC ENCRYPTION:\n')
    f.write('  Scheme: CKKS via TenSEAL (linear model proof-of-concept)\n')
    f.write(f'  Linear model accuracy: {he_linear_acc:.4f}\n')
    f.write(f'  Encryption time: {he_encrypt_time_ms:.2f} ms/sample\n')
    f.write(f'  Decryption time: {he_decrypt_time_ms:.2f} ms/sample\n')
    f.write('  Note: CNN inference would be 100-1000x slower\n')
    f.write('  Reference: Gilad-Bachrach et al. (2016) CryptoNets\n\n')
    
    f.write('CONCLUSION:\n')
    f.write('  - Differential Privacy: strong guarantees, 15-55% accuracy loss\n')
    f.write('  - Federated Learning: minimal accuracy loss, requires distributed setup\n')
    f.write('  - Homomorphic Encryption: strongest privacy, highest computational cost\n')

print('\nAll results saved to results/all_results.txt')
print('\n' + '=' * 60)
print('You now have all results ready for your paper!')
print('=' * 60)