import json
import zipfile
from tqdm import tqdm
import numpy as np
import h5py


def zip_to_hdf5_with_index(zip_filename, h5_filename, json_filename):
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref, h5py.File(h5_filename, 'w') as group:
        for file_path in tqdm(zip_ref.namelist()):
            if file_path.endswith('.png'):
                # Process as binary data
                with zip_ref.open(file_path, "r") as f:
                    binary_data = f.read()
                group.create_dataset(file_path, data=np.array(binary_data, dtype='S'))

            elif file_path.endswith('.json'):
                # Process as JSON
                with zip_ref.open(file_path, 'r') as f:
                    json_data = json.load(f)
                # Convert to string for storage
                json_str = json.dumps(json_data)
                json_bytes = json_str.encode('utf-8')
                group.create_dataset(file_path, data=np.array(json_bytes, dtype='S'))
            
            elif file_path.endswith('.npy'):
                with zip_ref.open(file_path, 'r') as f:
                    npy_data = np.load(f)
                group.create_dataset(file_path, data=npy_data)

            else:
                raise ValueError(f'Unknown file type: {file_path}')
        
        with open(json_filename, 'w') as f:
            # Save the json, so we know all the filenames in the h5
            json.dump(zip_ref.namelist(), f, indent=2)


zip_to_hdf5_with_index("data/images.zip", "data/images.h5", "data/images_h5.json")
zip_to_hdf5_with_index("data/vae-sd.zip", "data/vae-sd.h5", "data/vae-sd_h5.json")
