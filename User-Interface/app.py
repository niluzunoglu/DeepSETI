import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(
    page_title="DeepSETI | Sinyal Analiz Platformu",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def create_fake_database(num_signals=50):
    """Keşif için sahte bir sinyal veritabanı oluşturur."""
    np.random.seed(42)
    data = {
        'Sinyal ID': [f'DS_{i:04d}' for i in range(num_signals)],
        'Sinyal-Gürültü Oranı (SNR)': np.random.uniform(5.0, 50.0, num_signals).round(2),
        'Doppler Kayması (Hz/s)': np.random.uniform(-10.0, 10.0, num_signals).round(2),
        'RA': [f'19h {np.random.randint(0, 60)}m {np.random.randint(0, 60)}s' for _ in range(num_signals)],
        'DEC': [f'+{np.random.randint(0, 90)}d {np.random.randint(0, 60)}m {np.random.randint(0, 60)}s' for _ in range(num_signals)],
        'Durum': np.random.choice(['İncelenmedi', 'Aday Sinyal', 'Gürültü'], num_signals, p=[0.7, 0.2, 0.1])
    }
    return pd.DataFrame(data)

def generate_spectrogram_figure(signal_id):
    """Verilen sinyal ID'sine göre sahte bir spektrogram (şelale plotu) oluşturur."""
    np.random.seed(hash(signal_id) % (2**32 - 1))
    data = np.random.randn(256, 512) * 0.5 + 5
    start_row = np.random.randint(50, 200)
    drift = np.random.uniform(-0.5, 0.5)
    for i in range(512):
        row_idx = int(start_row + i * drift)
        if 0 <= row_idx < 256:
            data[row_idx, i] = np.random.uniform(20, 40) # Sinyal gücü
            
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale='viridis',
        colorbar=dict(title='Güç (dB)')
    ))
    fig.update_layout(
        title=f'<b>{signal_id} Zaman-Frekans Spektrogramı</b>',
        xaxis_title='Zaman Örneklemi',
        yaxis_title='Frekans Kanalı',
        height=500
    )
    return fig

def run_analysis_pipeline(pipeline_type, steps):
    """Seçilen pipeline adımlarını simüle eder ve sahte sonuçlar döndürür."""
    st.info(f"<b>{pipeline_type}</b> analiz akışı başlatılıyor...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_steps = len(steps)
    for i, step in enumerate(steps):
        status_text.text(f"Adım {i+1}/{total_steps}: {step} çalıştırılıyor...")
        time.sleep(np.random.uniform(1.5, 3.0)) # Adım süresini simüle et
        progress_bar.progress((i + 1) / total_steps)
        
    status_text.success("✅ Analiz Akışı Başarıyla Tamamlandı!")
    
    # Sahte performans metrikleri oluştur
    if pipeline_type == "Klasik ML (Random Forest, SVM)":
        metrics_data = {
            'Model': ['Random Forest', 'SVM'],
            'Precision': [np.random.uniform(0.85, 0.95), np.random.uniform(0.80, 0.90)],
            'Recall': [np.random.uniform(0.88, 0.98), np.random.uniform(0.82, 0.92)],
            'F1-Score': [np.random.uniform(0.86, 0.96), np.random.uniform(0.81, 0.91)]
        }
    else: # Derin Öğrenme (CNN)
        metrics_data = {
            'Model': ['CNN'],
            'Precision': [np.random.uniform(0.92, 0.99)],
            'Recall': [np.random.uniform(0.94, 0.99)],
            'F1-Score': [np.random.uniform(0.93, 0.99)]
        }
    
    metrics_df = pd.DataFrame(metrics_data).round(3)
    
    # Sahte karmaşıklık matrisi oluştur
    cm = np.random.randint(0, 100, size=(2,2))
    cm_labels = ['Gürültü', 'Sinyal']
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm, x=cm_labels, y=cm_labels,
        text=cm, texttemplate="%{text}", textfont={"size":20},
        colorscale='Blues'
    ))
    cm_fig.update_layout(title='<b>Karmaşıklık Matrisi</b>', xaxis_title='Tahmin Edilen Sınıf', yaxis_title='Gerçek Sınıf')

    return metrics_df, cm_fig


# --- Ana Arayüz ---

st.title("📡 DeepSETI: Teknolojik Sinyal Analiz Platformu")
st.markdown("""
Bu platform, SETI araştırmaları kapsamında toplanan radyo sinyallerini analiz etmek için modern makine öğrenmesi tekniklerini kullanır. 
Aşağıdaki araçları kullanarak potansiyel sinyal adaylarını keşfedebilir, analiz edebilir ve sınıflandırma modellerinin performansını değerlendirebilirsiniz.
""")

# Veritabanını yükle (simülasyon)
if 'signal_db' not in st.session_state:
    st.session_state.signal_db = create_fake_database()

# --- Kenar Çubuğu (Sidebar) ---
with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    
    st.subheader("1. Sinyal Filtreleme")
    snr_range = st.slider(
        "Sinyal-Gürültü Oranı (SNR) Aralığı",
        min_value=0.0, max_value=50.0, value=(15.0, 50.0)
    )
    drift_range = st.slider(
        "Doppler Kayması (Hz/s) Aralığı",
        min_value=-10.0, max_value=10.0, value=(-5.0, 5.0)
    )
    
    st.subheader("2. Pipeline Yönetimi")
    selected_pipeline = st.radio(
        "Analiz İş Akışı Seçin",
        ["Klasik ML (Random Forest, SVM)", "Derin Öğrenme (CNN)"]
    )
    
    st.write("**Çalıştırılacak Adımlar:**")
    run_preprocessing = st.checkbox("Sinyal Ön İşleme", value=True)
    run_feature_extraction = st.checkbox("Özellik Çıkarımı", value=True)
    run_model_training = st.checkbox("Model Eğitimi", value=True)
    
    if st.button("🚀 Analiz Akışını Başlat", type="primary"):
        st.session_state.run_analysis = True
    else:
        # Butona basılmadığında analiz durumunu sıfırla
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False
            
    st.markdown("---")
    st.info("Proje, Test Güdümlü Geliştirme (TDD) prensiplerine uygun olarak geliştirilmiştir.")


# --- Ana İçerik Alanı ---

# Filtrelenmiş veriyi oluştur
filtered_df = st.session_state.signal_db[
    (st.session_state.signal_db['Sinyal-Gürültü Oranı (SNR)'] >= snr_range[0]) &
    (st.session_state.signal_db['Sinyal-Gürültü Oranı (SNR)'] <= snr_range[1]) &
    (st.session_state.signal_db['Doppler Kayması (Hz/s)'] >= drift_range[0]) &
    (st.session_state.signal_db['Doppler Kayması (Hz/s)'] <= drift_range[1])
]

# 1. BÖLÜM: SİNYAL KEŞFİ VE SEÇİMİ
st.header("1. Sinyal Keşfi ve Görselleştirme")
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Filtrelenmiş Sinyaller")
    if filtered_df.empty:
        st.warning("Seçilen filtre kriterlerine uygun sinyal bulunamadı.")
    else:
        st.dataframe(filtered_df, height=500, use_container_width=True)
        
with col2:
    st.subheader("Sinyal Seçimi ve Spektrogram")
    if not filtered_df.empty:
        selected_signal_id = st.selectbox(
            "Görselleştirmek için bir sinyal seçin:",
            options=filtered_df['Sinyal ID']
        )
        if selected_signal_id:
            # Seçilen sinyalin spektrogramını oluştur ve göster
            spectrogram_fig = generate_spectrogram_figure(selected_signal_id)
            st.plotly_chart(spectrogram_fig, use_container_width=True)
    else:
        st.info("Görselleştirmek için lütfen sol tablodan bir sinyal seçin.")
        
st.markdown("---")

# 2. BÖLÜM: ANALİZ VE SONUÇLAR
st.header("2. Analiz Akışı Sonuçları")

if st.session_state.get('run_analysis', False):
    pipeline_steps_to_run = []
    if run_preprocessing: pipeline_steps_to_run.append("Ön İşleme")
    if run_feature_extraction: pipeline_steps_to_run.append("Özellik Çıkarımı")
    if run_model_training: pipeline_steps_to_run.append("Model Eğitimi")
    
    if not pipeline_steps_to_run:
        st.error("Lütfen çalıştırmak için en az bir pipeline adımı seçin.")
    else:
        # Analizi çalıştır ve sonuçları al (simülasyon)
        metrics_df, cm_fig = run_analysis_pipeline(selected_pipeline, pipeline_steps_to_run)
        
        # Sonuçları sekmeler halinde göster
        tab1, tab2, tab3 = st.tabs(["📊 Performans Metrikleri", "🔀 Karmaşıklık Matrisi", "📈 Karşılaştırmalı Grafik"])
        
        with tab1:
            st.subheader(f"{selected_pipeline} Modelleri İçin Performans")
            st.dataframe(metrics_df, use_container_width=True)
            
        with tab2:
            st.plotly_chart(cm_fig, use_container_width=True)
        
        with tab3:
            st.subheader("Modellerin F1-Skor Karşılaştırması")
            bar_fig = go.Figure(data=[
                go.Bar(name='F1-Score', x=metrics_df['Model'], y=metrics_df['F1-Score'], text=metrics_df['F1-Score'], textposition='auto')
            ])
            bar_fig.update_layout(yaxis_title='F1-Score', xaxis_title='Model')
            st.plotly_chart(bar_fig, use_container_width=True)

    # Analiz bittikten sonra tekrar çalışmaması için durumu sıfırla
    st.session_state.run_analysis = False
else:
    st.info("Bir analiz akışı çalıştırmak için kenar çubuğundaki **'Analiz Akışını Başlat'** düğmesine tıklayın.")