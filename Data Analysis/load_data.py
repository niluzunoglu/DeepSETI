import os
import pandas as pd
import h5py

def load_csv_data(file_path):
    """CSV formatındaki veriyi yükler."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Yüklendi: {file_path} - Satır: {df.shape[0]}, Sütun: {df.shape[1]}")
    return df

def load_hdf5_data(file_path, dataset_name):
    """HDF5 formatındaki veriyi yükler."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset bulunamadı: {dataset_name}")
        
        data = f[dataset_name][:]
        print(f"Yüklendi: {file_path} - Dataset: {dataset_name} - Şekil: {data.shape}")
        return data

if __name__ == "__main__":
    # Örnek kullanım
    csv_file = "../data/sample_data.csv"
    hdf5_file = "../data/sample_data.h5"
    dataset = "radio_signals"
    
    csv_data = load_csv_data(csv_file)
    hdf5_data = load_hdf5_data(hdf5_file, dataset)
