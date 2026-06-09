import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import requests

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="Skrinieng Kesehatan Payudara", page_icon="🎗️", layout="centered")

# ==========================================
# 2. FUNGSI MEMUAT MODEL PYTORCH (DARI HUGGING FACE)
# ==========================================
MODEL_URL = "https://huggingface.co/Dustingimaking1/BCancer/resolve/main/effnetb2_best.pth"
MODEL_PATH = "effnetb2_best.pth"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mempersiapkan sistem (mengunduh model sekitar 32 MB)..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

    # Load arsitektur dasar EfficientNet-B2
    model = models.efficientnet_b2(weights=None)
    
    # Rekonstruksi classifier
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True), 
        nn.Linear(num_ftrs, 128),        
        nn.ReLU(),                       
        nn.Dropout(p=0.3),               
        nn.Linear(128, 2)                
    )
    
    # Load bobot model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Panggil fungsi agar model dimuat
model = load_my_model()

# ==========================================
# 3. PREPROCESSING GAMBAR
# ==========================================
transform = transforms.Compose([
    transforms.Resize((288, 288)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. INJEKSI CSS KUSTOM (UKURAN DIKECILKAN & PROPORSIONAL)
# ==========================================
custom_css = """
<style>
    /* Latar belakang aplikasi */
    .stApp { background-color: #fbe6eb; }
    
    /* Memusatkan wadah gambar Streamlit */
    [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"] {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin-left: auto;
        margin-right: auto;
    }

    /* Mengunci ukuran gambar tetap di tengah (300x300) */
    [data-testid="stImage"] img {
        width: 300px !important;
        height: 300px !important;
        object-fit: cover !important;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin: 0 !important;
        display: block;
    }

    /* Sembunyikan label bawaan file uploader */
    .stFileUploader > label {
        display: none;
    }

    /* Pengaturan teks sambutan (Dikecilkan dari 24px ke kisaran 15px-18px) */
    .welcome-text { 
        color: #c24068; 
        text-align: center; 
        font-size: clamp(14px, 3.5vw, 18px); 
        font-weight: 600; 
        margin-bottom: 4%; 
        padding: 0 5%; 
        line-height: 1.5; 
    }
    
    /* Teks Disclaimer (Dikecilkan menjadi lebih subtil) */
    .disclaimer-text { 
        color: #e91e63; 
        text-align: center; 
        font-size: clamp(11px, 2.5vw, 13px); 
        font-weight: 500; 
        margin-top: 6%; 
        padding: 0 8%; 
        line-height: 1.6; 
        opacity: 0.85;
    }
    
    /* Kotak Hasil Prediksi (Lebar disamakan dengan gambar agar rapi) */
    .result-box { 
        border-radius: 12px; 
        padding: 12px 20px; 
        text-align: center; 
        margin-top: 4%; 
        box-shadow: 0px 3px 6px rgba(0,0,0,0.05); 
        width: 300px; /* Lebarnya disamakan persis dengan ukuran gambar */
        margin-left: auto; 
        margin-right: auto; 
    }
    
    /* Teks di dalam Kotak Hasil (Dikecilkan agar pas di dalam kotak 300px) */
    .result-text { 
        font-size: clamp(16px, 4vw, 20px); 
        font-weight: bold; 
        margin: 0; 
    }
    
    /* Kontainer dan ukuran Icon Ribbon di atas (Diperkecil) */
    .logo-container { display: flex; justify-content: center; margin-top: 5%; margin-bottom: 2%; }
    .ribbon-icon { 
        background-color: #f8bbd0; 
        color: #e91e63; 
        font-size: clamp(24px, 5vw, 30px); 
        width: clamp(50px, 12vw, 60px); 
        height: clamp(50px, 12vw, 60px); 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        border-radius: 50%; 
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 5. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.markdown('<div class="logo-container"><div class="ribbon-icon">🎗️</div></div>', unsafe_allow_html=True)
st.markdown('<div class="welcome-text">Selamaet datang! Mari kita langkah bersama untuk mengecek kesehatan payudara Anda<br>dengan penuh kepedulian dan kehangatan—selangkah demi selangkah.</div>', unsafe_allow_html=True)

# File uploader
with st.container():
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    # 1. Tampilkan Gambar (Center via CSS)
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image)
    
    # 2. Proses Prediksi PyTorch
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    
    # 3. Logika Hasil Prediksi (0 = Jinak, 1 = Ganas)
    if predicted_class == 1:
        hasil_teks = "Ganas"
        warna_box = "#ffebee"  # Merah muda
        warna_teks = "#c62828" # Merah tua
    else:
        hasil_teks = "Jinak"
        warna_box = "#e0f2f1"  # Hijau muda
        warna_teks = "#2e7d32" # Hijau tua
        
    # 4. Tampilkan Kotak Hasil
    st.markdown(
        f'<div class="result-box" style="background-color: {warna_box};"><p class="result-text" style="color: {warna_teks};">Hasil Prediksi : {hasil_teks}</p></div>', 
        unsafe_allow_html=True
    )

st.markdown('<div class="disclaimer-text">Mengingat ini adalah alat skrining dan edukasi, kami sangat menyarankan Anda untuk tetap berkonsultasi<br>dengan dokter spesialis demi mendapatkan diagnosis medis yang menyeluruh.</div>', unsafe_allow_html=True)
