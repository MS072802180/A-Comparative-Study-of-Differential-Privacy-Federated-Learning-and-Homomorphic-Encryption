import pickle
import numpy as np

print("Loading small dataset...")
with open('data/small_dataset.pkl', 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y']

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Y values: {np.unique(Y)}")
print(f"X data type: {X.dtype}")
print(f"X min: {X.min():.4f}, X max: {X.max():.4f}")