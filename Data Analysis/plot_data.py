import blimpy as bl
import matplotlib.pyplot as plt

class WaterfallAnalyzer:
    def __init__(self, file_path):
        """HDF5 dosyasını yükleyerek bir Waterfall nesnesi oluşturur."""
        self.file_path = file_path
        self.wf = None  # Waterfall nesnesi
        
    def load_data(self):
        """Waterfall nesnesini oluştur ve veriyi yükle."""
        try:
            self.wf = bl.Waterfall(self.file_path, load_data=True)
            print(f"✅ Veri başarıyla yüklendi: {self.file_path}")
            print(f"🔹 Veri şekli: {self.wf.data.shape}")
            print(f"🔹 Header bilgileri: {self.wf.header}")
        except Exception as e:
            print(f"❌ Veri yüklenirken hata oluştu: {e}")

    def plot_waterfall(self, save_path=None):
        """Waterfall grafiğini çiz ve isteğe bağlı olarak kaydet."""
        if self.wf is None:
            print("❌ Veri yüklenmemiş! Lütfen önce load_data() metodunu çağırın.")
            return

        plt.figure(figsize=(10, 6))
        self.wf.plot_waterfall()
        plt.xlabel('Frekans (MHz)')
        plt.ylabel('Zaman (s)')
        plt.title('Waterfall Plot')

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"✅ Waterfall plot '{save_path}' olarak kaydedildi!")
        else:
            plt.show()

# Kullanım
file_path = "../Sample Data/Parkes_58135_06209_ALPHACEN_mid.h5"
analyzer = WaterfallAnalyzer(file_path)
analyzer.load_data()
analyzer.plot_waterfall("plots/alphacen_waterfall_plot.png")  # Grafiği kaydet
