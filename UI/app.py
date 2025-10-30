import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import setigen as stg


# --- Veri Yükleme Fonksiyonları ---

@st.cache_data(show_spinner="Büyük veriyi yükleniyor ve ön işleniyor...")
def load_and_preprocess_data(data_index):
    """
    setigen.Frame oluşturulması - DÜZELTME: Doğru parametre kullanımı
    """
    st.write(f"⚙️ Veri Yükleniyor: Parça #{data_index} (Sadece bir kez çalışır)")

    # Doğru parametre formatı ile Frame oluştur
    frame = stg.Frame(
        fchans=1024,  # Frekans kanalı sayısı
        tchans=512,  # Zaman kanalı sayısı
        df=2.7939677238464355e-06,  # Frekans çözünürlüğü (MHz)
        dt=18.253611008,  # Zaman çözünürlüğü (saniye)
        fch1=6095.214842353016  # Başlangıç frekansı (MHz)
    )

    # Gaussian gürültü ekle
    frame.add_noise(x_mean=1, x_std=5, noise_type='gaussian')

    return frame.data  # NumPy array döndür


@st.cache_data(show_spinner="Spektrogram Down-Sampling yapılıyor...")
def downsample_for_visualization(data_array, block_size):
    """
    Down-sampling (Blok Ortalama) işlemi
    """
    if block_size <= 1:
        return data_array

    # Boyutları al
    h, w = data_array.shape

    # Yeni boyutları hesapla
    new_h = h // block_size
    new_w = w // block_size

    # Reshape ve ortalama al
    downsampled = data_array[:new_h * block_size, :new_w * block_size].reshape(
        new_h, block_size, new_w, block_size
    ).mean(axis=(1, 3))

    return downsampled


# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="DeepSETI Etkileşimli Analiz Platformu MVP")

st.title("📡 DeepSETI Etkileşimli Sinyal Analiz Platformu")
st.markdown("---")

# Session State
if 'current_data_index' not in st.session_state:
    st.session_state.current_data_index = 1
if 'anomaly_threshold' not in st.session_state:
    st.session_state.anomaly_threshold = 10.0
if 'signal_inject_freq' not in st.session_state:
    st.session_state.signal_inject_freq = 0.5

# Sidebar Kontrol Paneli
with st.sidebar:
    st.header("🔬 Kontrol Paneli")
    st.markdown("Bu, **setigen** parametrelerini ve deteksiyon ayarlarını kontrol ettiğiniz alandır.")

    # Veri Yükleme
    st.subheader("1. Veri Kaynağı")
    data_select = st.selectbox(
        "Yüklenecek Spektrogram Parçasını Seçin (Simülasyon)",
        [1, 2, 3],
        index=st.session_state.current_data_index - 1
    )
    if st.session_state.current_data_index != data_select:
        st.session_state.current_data_index = data_select
        st.rerun()

    # Sinyal Enjeksiyonu
    st.subheader("2. 🧪 Sentetik Enjeksiyon Sandbox")
    st.markdown("Sinyal parametrelerini ayarlayın ve gerçek zamanlı sinyal enjekte edin.")

    enable_injection = st.checkbox("Sentetik Sinyal Enjeksiyonunu Etkinleştir", value=True)

    inject_freq = st.slider(
        "Enjeksiyon Frekansı (0.0 - 1.0)",
        0.0, 1.0, st.session_state.signal_inject_freq, 0.01,
        key='inject_freq_slider',
        help="Sinyalin spektrogram üzerindeki dikey konumunu belirler."
    )
    st.session_state.signal_inject_freq = inject_freq

    inject_drift = st.number_input(
        "Frekans Kayması (Drift Rate)",
        -0.01, 0.01, 0.0001, 0.0001,
        help="Sinyalin zamanla ne kadar kaydığını ayarlar."
    )

    inject_power = st.number_input(
        "Sinyal Gücü (Power)",
        10, 200, 100, 10
    )

    # Deteksiyon Ayarları
    st.subheader("3. 🎯 Anomali Deteksiyon Ayarları")
    anomaly_threshold = st.slider(
        "Anomali Eşik Değeri (Reconstruction Error Cutoff)",
        0.0, 50.0, st.session_state.anomaly_threshold, 1.0,
        key='threshold_slider'
    )
    st.session_state.anomaly_threshold = anomaly_threshold

# --- Ana İşlem Akışı ---

# Temel veriyi yükle
base_data = load_and_preprocess_data(st.session_state.current_data_index)

# Kopyasını al
processed_data = np.copy(base_data)

# Sinyal Enjeksiyonu - DÜZELTME: setigen Frame yerine manuel NumPy işlemi
if enable_injection:
    with st.spinner("Sentetik sinyal enjekte ediliyor..."):

        freq_channels, time_channels = processed_data.shape

        # Sinyal merkez frekansı (pixel indeksi)
        center_freq_idx = int(inject_freq * freq_channels)
        signal_width = 4  # Sinyal genişliği (piksel)

        # Gaussian sinyal profili oluştur
        freq_indices = np.arange(freq_channels)
        gaussian_profile = np.exp(
            -((freq_indices - center_freq_idx) ** 2) / (2 * signal_width ** 2)
        )

        # Her zaman kanalına sinyal ekle (drift ile)
        for t_idx in range(time_channels):
            # Drift efekti (zamanla frekans kayması)
            drift_offset = int(inject_drift * 10000 * (t_idx - time_channels / 2))
            shifted_center = center_freq_idx + drift_offset

            # Sınırlar içinde mi kontrol et
            if 0 <= shifted_center < freq_channels:
                # Sinyal ekle
                for f_offset in range(-signal_width * 2, signal_width * 2 + 1):
                    f_idx = shifted_center + f_offset
                    if 0 <= f_idx < freq_channels:
                        signal_amplitude = inject_power * gaussian_profile[f_idx]
                        processed_data[f_idx, t_idx] += signal_amplitude

# Down-sampling (Görselleştirme için)
DOWN_SAMPLE_BLOCK_SIZE = 4
plot_data = downsample_for_visualization(processed_data, DOWN_SAMPLE_BLOCK_SIZE)

# --- Görselleştirme ---

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🖼️ Dinamik Spektrogram ve Anomali Haritası")
    st.markdown(
        f"**Görüntülenen Çözünürlük:** {plot_data.shape[1]} x {plot_data.shape[0]} "
        f"(Orijinal: {base_data.shape} - Blok Ortalama: {DOWN_SAMPLE_BLOCK_SIZE}x)"
    )

    # Matplotlib plot
    fig, ax = plt.subplots(figsize=(12, 7))

    im = ax.imshow(
        plot_data,
        aspect='auto',
        interpolation='nearest',
        origin='lower',
        cmap='viridis'
    )

    ax.set_title(
        f"İşlenmiş Spektrogram (Gürültü + Sinyal Enjeksiyonu: {enable_injection})",
        fontsize=14,
        fontweight='bold'
    )
    ax.set_xlabel("Zaman Kanalları", fontsize=12)
    ax.set_ylabel("Frekans Kanalları", fontsize=12)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Güç (Arbitrary Units)', rotation=270, labelpad=20)

    # Sinyal pozisyonunu işaretle
    if enable_injection:
        y_loc = int(plot_data.shape[0] * inject_freq / DOWN_SAMPLE_BLOCK_SIZE)
        ax.axhline(
            y=y_loc,
            color='red',
            linestyle='--',
            linewidth=2.5,
            label='Enjekte Edilen Sinyal',
            alpha=0.8
        )
        ax.legend(loc='upper right', fontsize=10)

    st.pyplot(fig)

with col2:
    st.subheader("📊 Model Çıktıları")
    st.markdown("Reconstruction Error Map ve Anomali skorları")

    # XAI Açıklaması
    st.info(
        "**🧠 XAI Prensibi:** Autoencoder sadece gürültü üzerinde eğitilmiştir. "
        "Spektrogramdaki yapılandırılmış bir sinyal (**anomali**), yüksek bir "
        "**Yeniden Yapılandırma Hatası** skoruna neden olur."
    )

    # Anomali Skoru (Simülasyon)
    if enable_injection:
        # Sinyal var → Yüksek skor
        anomaly_score = np.random.uniform(
            st.session_state.anomaly_threshold + 5,
            st.session_state.anomaly_threshold + 20
        )
        st.metric(
            "Reconstruction Error (MAE)",
            f"{anomaly_score:.2f}",
            f"+{anomaly_score - st.session_state.anomaly_threshold:.2f} (Eşik Üstü)",
            delta_color="inverse"
        )

        if anomaly_score > st.session_state.anomaly_threshold:
            st.error("🚨 **ANOMALİ TESPİT EDİLDİ!**")
            st.markdown("**Olası Technosignature Candidate**")
        else:
            st.success("✅ Temiz Sinyal (Gürültü Normunda)")
    else:
        # Sinyal yok → Düşük skor
        anomaly_score = np.random.uniform(
            st.session_state.anomaly_threshold - 15,
            st.session_state.anomaly_threshold - 5
        )
        st.metric(
            "Reconstruction Error (MAE)",
            f"{anomaly_score:.2f}",
            f"{anomaly_score - st.session_state.anomaly_threshold:.2f} (Eşik Altı)",
            delta_color="normal"
        )
        st.success("✅ Temiz Sinyal (Gürültü Normunda)")

    st.markdown("---")

    # Detay Tablosu
    st.subheader("📍 Anomali Koordinatları")

    detail_df = pd.DataFrame({
        'Parametre': ['Zaman', 'Frekans (norm)', 'Error Skoru', 'SNR'],
        'Değer': [
            'T-0s',
            f"{inject_freq:.3f}",
            f"{anomaly_score:.2f}",
            f"{inject_power / 5:.1f}"
        ]
    })
    st.dataframe(detail_df, hide_index=True, use_container_width=True)

    # Ek Metrikler
    with st.expander("ℹ️ Veri İstatistikleri"):
        st.write(f"**Min Değer:** {processed_data.min():.2f}")
        st.write(f"**Max Değer:** {processed_data.max():.2f}")
        st.write(f"**Ortalama:** {processed_data.mean():.2f}")
        st.write(f"**Std Sapma:** {processed_data.std():.2f}")

st.markdown("---")
st.caption(
    "✅ **Streamlit Best Practices:** Caching (@st.cache_data), "
    "Session State, Component-Based Layout uygulanmıştır."
)
st.caption("🛸 DeepSETI MVP - Breakthrough Listen tarzı sinyal analiz simülasyonu")