import h5py

file_path = 'C:/Users/Admin/Desktop/privacy_modulation_project/data/RADIOML 2018.hdf5'

try:
    f = h5py.File(file_path, 'r')
    print("File opened successfully!")
    print(f.keys())
    f.close()
except Exception as e:
    print(f"Error: {e}")