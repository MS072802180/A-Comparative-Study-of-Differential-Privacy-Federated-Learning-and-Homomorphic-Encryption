import h5py
import numpy as np
import pickle

file_path = 'C:/Users/Admin/Desktop/privacy_modulation_project/data/RADIOML 2018.hdf5'

print("Opening full dataset...")
f = h5py.File(file_path, 'r')

X_full = f['X'][:]  # Signals
Y_full = f['Y'][:]  # Labels (one-hot)
Z_full = f['Z'][:]  # SNR values

print(f"Full dataset shape: X: {X_full.shape}, Y: {Y_full.shape}, Z: {Z_full.shape}")

# Find which SNR values are available
unique_snrs = np.unique(Z_full)
print(f"Available SNR values: {unique_snrs[:10]}... (showing first 10)")

# Find SNR = 10 dB (or closest)
target_snr = 10
snr_idx = np.where(np.abs(Z_full - target_snr) < 0.1)[0]
print(f"Found {len(snr_idx)} samples at SNR close to {target_snr} dB")

# Get data at target SNR
X_snr = X_full[snr_idx]
Y_snr = Y_full[snr_idx]

print(f"Data at SNR {target_snr}: X shape {X_snr.shape}, Y shape {Y_snr.shape}")

# Get modulation class names (we need to map one-hot to class indices)
# Y is one-hot encoded. Let's find which modulations are present
Y_indices = np.argmax(Y_snr, axis=1)
unique_classes = np.unique(Y_indices)
print(f"Modulation classes present: {unique_classes}")

# For now, let's just take the first 4 modulation classes
selected_classes = unique_classes[:4]
print(f"Selected classes: {selected_classes}")

# Create masks for selected classes
mask = np.isin(Y_indices, selected_classes)
X_filtered = X_snr[mask]
Y_filtered = Y_snr[mask]
Y_indices_filtered = Y_indices[mask]

print(f"After filtering classes: X shape {X_filtered.shape}")

# Take 5000 samples per class (or whatever is available)
samples_per_class = 5000
X_small = []
Y_small = []
for class_idx in selected_classes:
    class_mask = (Y_indices_filtered == class_idx)
    class_samples = X_filtered[class_mask]
    n_samples = min(samples_per_class, len(class_samples))
    X_small.append(class_samples[:n_samples])
    # Create new labels (0,1,2,3 for our 4 classes)
    Y_small.append(np.ones(n_samples) * class_idx)

X_small = np.concatenate(X_small, axis=0)
Y_small = np.concatenate(Y_small, axis=0)

print(f"Small dataset created: X shape {X_small.shape}, Y shape {Y_small.shape}")

# Save the small dataset
save_path = 'C:/Users/Admin/Desktop/privacy_modulation_project/data/small_dataset.pkl'
with open(save_path, 'wb') as file:
    pickle.dump({'X': X_small, 'Y': Y_small}, file)

print(f"Saved small dataset to {save_path}")

f.close()
print("Done!")