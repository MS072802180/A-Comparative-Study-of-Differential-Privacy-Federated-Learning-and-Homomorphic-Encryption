import h5py

# Use forward slashes to avoid escape character problems
file_path = 'C:/Users/Admin/Desktop/privacy_modulation_project/data/RADIOML 2018.hdf5'

print("Opening file...")
f = h5py.File(file_path, 'r')

print("\nKeys in the file:")
for key in f.keys():
    print(f"  {key}")

print("\nShapes of each dataset:")
for key in f.keys():
    try:
        print(f"  {key}: {f[key].shape}")
    except:
        print(f"  {key}: this is a group, not a dataset")

print("\nExploring all groups and datasets:")
def explore(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"  {name}: {obj.shape}")

f.visititems(explore)

f.close()
print("\nDone.")