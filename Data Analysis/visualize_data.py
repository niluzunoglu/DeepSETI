import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

def plot_csv_histogram(file_path, column_name):
    """CSV dosyasındaki belirli bir sütunun histogramını çizer."""
    df = pd.read_csv(file_path)
    if column_name not in df.columns:
        raise KeyError(f"Sütun bulunamadı: {column_name}")
    
    plt.figure(figsize=(8, 5))
    plt.hist(df[column_name].dropna(), bins=50, color='blue', alpha=0.7)
    plt.xlabel(column_name)
    plt.ylabel("Frekans")
    plt.title(f"{column_name} Histogramı")
    plt.grid(True)
    plt.show()

def plot_hdf5_signal(file_path, dataset_name, sample_index=0):
    """HDF5 dosyasındaki belirli bir sinyalin çizimini yapar."""
    with h5py.File(file_path, 'r') as f:
        if dataset_name not in f:
            raise KeyError(f"Dataset bulunamadı: {dataset_name}")
        
        data = f[dataset_name][:]
        if sample_index >= data.shape[0]:
            raise IndexError(f"Geçersiz indeks: {sample_index}, maksimum: {data.shape[0]-1}")
        
        signal = data[sample_index]
        plt.figure(figsize=(10, 4))
        plt.plot(signal, color='red')
        plt.xlabel("Zaman")
        plt.ylabel("Genlik")
        plt.title(f"{dataset_name} - Örnek {sample_index}")
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # Örnek kullanım
    csv_file = "../data/sample_data.csv"
    column = "signal_power"
    plot_csv_histogram(csv_file, column)
    
    hdf5_file = "../data/sample_data.h5"
    dataset = "radio_signals"
    plot_hdf5_signal(hdf5_file, dataset, sample_index=0)
