import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

os.makedirs('plots', exist_ok=True)

# Data from your HE_comparison.py
model_names = ['Logistic Regression\n(Linear)', 'MLP with x²\n(Nonlinear)']
model_params = [256, 4352]
plaintext_acc = [0.2850, 0.6900]
encrypted_acc = [0.2850, 0.6900]
inference_time = [75.83, 1516.13]  # ms
encrypt_time = 5.49  # ms

# Baseline CNN for reference
baseline_acc = 0.9112

print('=' * 60)
print('HE FIXED PLOT - Plaintext vs Encrypted Accuracy')
print('=' * 60)

# Create figure with 2 subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Homomorphic Encryption: Detailed Results', fontsize=14, fontweight='bold')

# ============================================================
# PLOT 1: Accuracy vs Model Complexity (Line plot like DP/FL)
# ============================================================
x_pos = np.arange(len(model_names))
width = 0.35

# Bar chart for clarity
bars1 = ax1.bar(x_pos - width/2, plaintext_acc, width, label='Plaintext Accuracy', 
                color='blue', alpha=0.7, edgecolor='black')
bars2 = ax1.bar(x_pos + width/2, encrypted_acc, width, label='Encrypted (HE) Accuracy', 
                color='orange', alpha=0.7, edgecolor='black')

# Add baseline line
ax1.axhline(y=baseline_acc, color='r', linestyle='--', linewidth=2, 
            label=f'CNN Baseline ({baseline_acc:.3f})')

ax1.set_xlabel('Model Type', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Homomorphic Encryption: Plaintext vs Encrypted Accuracy', fontsize=12)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(model_names)
ax1.set_ylim([0, 1.1])
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}', 
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', fontsize=9, fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax1.annotate(f'{height:.3f}', 
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3), textcoords="offset points",
                ha='center', fontsize=9, fontweight='bold')

# ============================================================
# PLOT 2: Inference Time Comparison
# ============================================================
bars3 = ax2.bar(model_names, inference_time, color=['orange', 'red'], alpha=0.7, edgecolor='black')
ax2.set_xlabel('Model Type', fontsize=12)
ax2.set_ylabel('Inference Time (ms) - Log Scale', fontsize=12)
ax2.set_title('Homomorphic Encryption: Inference Time Overhead', fontsize=12)
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars3, inference_time):
    ax2.annotate(f'{val:.1f} ms', 
                xy=(bar.get_x() + bar.get_width()/2, val),
                xytext=(0, 5), textcoords="offset points",
                ha='center', fontsize=9, fontweight='bold')

# Add annotation for plaintext inference time
ax2.annotate('Plaintext inference: ~0.05 ms', 
            xy=(0.5, 0.5), xycoords='axes fraction',
            fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('plots/he_fixed_plot.png', dpi=150, bbox_inches='tight')
print('Plot saved to plots/he_fixed_plot.png')

# ============================================================
# CREATE A LINE PLOT (like DP and FL) FOR HE
# ============================================================
fig2, ax3 = plt.subplots(figsize=(8, 6))

# Plot plaintext and encrypted as lines vs model complexity
x_vals = [1, 2]  # 1 for Linear, 2 for Nonlinear
ax3.plot(x_vals, plaintext_acc, 'b-o', linewidth=2, markersize=10, 
         label='Plaintext Accuracy', markerfacecolor='white', markeredgewidth=2)
ax3.plot(x_vals, encrypted_acc, 'r-s', linewidth=2, markersize=10, 
         label='Encrypted (HE) Accuracy', markerfacecolor='white', markeredgewidth=2)

# Add baseline line
ax3.axhline(y=baseline_acc, color='g', linestyle='--', linewidth=2, 
            label=f'CNN Baseline ({baseline_acc:.3f})')

ax3.set_xlabel('Model Complexity (1=Linear, 2=Nonlinear MLP)', fontsize=12)
ax3.set_ylabel('Accuracy', fontsize=12)
ax3.set_title('HE: Accuracy vs Model Complexity (Line Plot Style)', fontsize=14)
ax3.set_xticks([1, 2])
ax3.set_xticklabels(['Linear\n(256 params)', 'Nonlinear\n(4352 params)'])
ax3.set_ylim([0, 1])
ax3.grid(True, alpha=0.3)
ax3.legend(loc='lower right')

# Add value labels
for i, (x, y) in enumerate(zip(x_vals, plaintext_acc)):
    ax3.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=9)
for i, (x, y) in enumerate(zip(x_vals, encrypted_acc)):
    ax3.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(5, -10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('plots/he_line_plot.png', dpi=150, bbox_inches='tight')
print('Line plot saved to plots/he_line_plot.png')

# ============================================================
# PRINT SUMMARY
# ============================================================
print('\n' + '=' * 60)
print('FIXED HE RESULTS SUMMARY')
print('=' * 60)
print(f'\nModel 1: Logistic Regression (Linear)')
print(f'  Plaintext Accuracy:  {plaintext_acc[0]:.4f} ({plaintext_acc[0]*100:.1f}%)')
print(f'  Encrypted Accuracy:  {encrypted_acc[0]:.4f} ({encrypted_acc[0]*100:.1f}%)')
print(f'  Inference Time:      {inference_time[0]:.2f} ms')
print(f'  Parameters:          {model_params[0]}')
print(f'  Slowdown:            {inference_time[0]/0.05:.0f}x')

print(f'\nModel 2: MLP with x² Activation (Nonlinear)')
print(f'  Plaintext Accuracy:  {plaintext_acc[1]:.4f} ({plaintext_acc[1]*100:.1f}%)')
print(f'  Encrypted Accuracy:  {encrypted_acc[1]:.4f} ({encrypted_acc[1]*100:.1f}%)')
print(f'  Inference Time:      {inference_time[1]:.2f} ms')
print(f'  Parameters:          {model_params[1]}')
print(f'  Slowdown:            {inference_time[1]/0.05:.0f}x')

print(f'\nCNN Baseline Accuracy: {baseline_acc:.4f} ({baseline_acc*100:.1f}%)')
print(f'Encryption Time (both models): {encrypt_time:.2f} ms/sample')

print('\nFiles generated:')
print('  - plots/he_fixed_plot.png (bar chart comparison)')
print('  - plots/he_line_plot.png (line plot like DP/FL)')
print('=' * 60)