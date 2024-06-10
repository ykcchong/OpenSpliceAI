import h5py

def explore_hdf5_file(file_path):
    """
    Function to explore the contents of an HDF5 file.
    
    Parameters:
    file_path (str): Path to the HDF5 file.
    """
    # Open the HDF5 file
    with h5py.File(file_path, 'r') as hdf:
        print("Exploring the structure of the HDF5 file:")
        
        def print_attrs(name, obj):
            """ Helper function to print the attributes of a dataset/group """
            print(f"\nName: {name}")
            print("Type:", "Group" if isinstance(obj, h5py.Group) else "Dataset")
            print("Attributes:")
            for key, val in obj.attrs.items():
                print(f"    {key}: {val}")
        
        # Traverse and print the structure
        hdf.visititems(print_attrs)

        print("\nExploring datasets and their contents:")
        def explore_dataset(name, obj):
            """ Helper function to explore datasets """
            if isinstance(obj, h5py.Dataset):
                print(f"\nDataset: {name}")
                print("Shape:", obj.shape)
                print("Data type:", obj.dtype)
                print("Sample data:")
                # Print a small sample of the dataset
                data_sample = obj[...]
                print(data_sample[:10] if data_sample.size > 10 else data_sample)
        
        # Traverse and print dataset information
        hdf.visititems(explore_dataset)

# Replace 'your_file.h5' with the path to your HDF5 file
folder_path = '/ccb/cybertron/smao10/openspliceai/results/predict'
file_path = f'{folder_path}/SpliceAI_5000_400_3_45/predict.h5'
explore_hdf5_file(file_path)