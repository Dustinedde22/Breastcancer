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
st.set_page_config(page_title="Skrining Kesehatan Payudara", page_icon="🎗️", layout="centered")

# ==========================================
# 2. FUNGSI MEMUAT MODEL PYTORCH (DARI HUGGING FACE)
# ==========================================
# Tautan langsung dari Hugging Face
MODEL_URL = "https://huggingface.co/Dustingimaking1/BCancer/resolve/main/effnetb2_best.pth"
MODEL_PATH = "effnetb2_best.pth"

@st.cache_resource
def load_my_model():
    # --- PROSES DOWNLOAD MODEL JIKA BELUM ADA ---
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mempersiapkan sistem untuk pertama kali (mengunduh model sekitar 32 MB)..."):
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)

    # --- PROSES LOAD ARSITEKTUR ---
    # Panggil arsitektur dasar EfficientNet-B2
    model = models.efficientnet_b2(weights=None)
    
    # Ambil jumlah fitur input (1408)
    num_ftrs = model.classifier[1].in_features
    
    # Rekonstruksi classifier persis seperti saat model ditraining
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True), 
        nn.Linear(num_ftrs, 128),        
        nn.ReLU(),                       
        nn.Dropout(p=0.3),               
        nn.Linear(128, 2)                
    )
    
    # Muat bobot model dengan aman ke CPU dari file yang baru diunduh
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

# Panggil fungsi agar model dimuat ke memori
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
# 4. INJEKSI CSS KUSTOM (RESPONSIF & GAMBAR FIXED)
# ==========================================
custom_css = """
<style>
    .stApp { background-color: #fbe6eb; }
    
    /* Memusatkan wadah gambar */
    [data-testid="stImage"] { display: flex; justify-content: center; margin: 5% 0; }
    
    /* MENGUNCI UKURAN GAMBAR (FIXED SIZE) */
    [data-testid="stImage"] img { 
        width: 300px !important;       /* Lebar dikunci tetap */
        height: 300px !important;      /* Tinggi dikunci tetap */
        object-fit: cover !important;  /* Memotong proporsional agar tidak gepeng */
        border-radius: 15px;           /* Ujung gambar dibuat membulat */
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1); /* Sedikit bayangan agar elegan */
    }
    
    .welcome-text { color: #c24068; text-align: center; font-size: clamp(18px, 4vw, 24px); font-weight: 600; margin-bottom: 5%; padding: 0 3%; line-height: 1.5; }
    .disclaimer-text { color: #e91e63; text-align: center; font-size: clamp(14px, 3vw, 18px); font-weight: 500; margin-top: 8%; padding: 0 5%; line-height: 1.5; }
    
    .result-box { border-radius: 15px; padding: 5%; text-align: center; margin-top: 5%; box-shadow: 0px 4px 6px rgba(0,0,0,0.05); width: 80%; margin-left: auto; margin-right: auto; }
    .result-text { font-size: clamp(24px, 5vw, 32px); font-weight: bold; margin: 0; }
    
    .logo-container { display: flex; justify-content: center; margin-top: 8%; margin-bottom: 3%; }
    .ribbon-icon { background-color: #f8bbd0; color: #e91e63; font-size: clamp(40px, 8vw, 50px); width: clamp(80px, 20vw, 100px); height: clamp(80px, 20vw, 100px); display: flex; align-items: center; justify-content: center; border-radius: 50%; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ==========================================
# 5. ANTARMUKA PENGGUNA (UI)
# ==========================================
st.markdown('<div class="logo-container"><div class="ribbon-icon">🎗️</div></div>', unsafe_allow_html=True)
st.markdown('<div class="welcome-text">Selamat datang! Mari kita langkah bersama untuk mengecek kesehatan payudara Anda<br>dengan penuh kepedulian dan kehangatan—selangkah demi selangkah.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    # 1. Tampilkan Gambar
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, use_container_width=True)
    
    # 2. Proses Prediksi PyTorch
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    
    # 3. Logika Hasil Prediksi (Asumsi: 0 = Jinak, 1 = Ganas)
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