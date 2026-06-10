import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import os
import requests
import base64
from io import BytesIO

st.set_page_config(page_title="Skrining Kesehatan Payudara", page_icon="🎗️", layout="centered")

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
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_my_model()

transform = transforms.Compose([
    transforms.Resize((288, 288)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# FUNGSI KUSTOM: Mengubah PIL Image ke format HTML Base64 agar bisa di-center sempurna
def get_image_download_link(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f'<div class="img-container"><img src="data:image/jpeg;base64,{img_str}" class="center-img"></div>'

custom_css = """
<style>
    /* Latar belakang aplikasi */
    .stApp { background-color: #fbe6eb; }
    
    /* Sembunyikan label bawaan file uploader */
    .stFileUploader > label {
        display: none;
    }

    /* Pembungkus Gambar agar berada tepat di tengah halaman */
    .img-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin: 20px 0;
    }

    /* Mengatur dan mengecilkan ukuran gambar (Sekarang 200px agar tidak kebesaran) */
    .center-img {
        width: 200px !important;
        height: 200px !important;
        object-fit: cover !important;
        border-radius: 12px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }

    /* Pengaturan teks sambutan */
    .welcome-text { 
        color: #c24068; 
        text-align: center; 
        font-size: 16px; 
        font-weight: 600; 
        margin-bottom: 25px; 
        padding: 0 5%; 
        line-height: 1.5; 
    }
    
    /* Teks Disclaimer */
    .disclaimer-text { 
        color: #e91e63; 
        text-align: center; 
        font-size: 12px; 
        font-weight: 500; 
        margin-top: 35px; 
        padding: 0 8%; 
        line-height: 1.6; 
        opacity: 0.85;
    }
    
    /* Kotak Hasil Prediksi (Lebar disamakan dengan ukuran gambar baru yaitu 200px) */
    .result-box { 
        border-radius: 10px; 
        padding: 10px 15px; 
        text-align: center; 
        margin-top: 20px; 
        box-shadow: 0px 3px 6px rgba(0,0,0,0.05); 
        width: 200px; 
        margin-left: auto; 
        margin-right: auto; 
    }
    
    /* Teks di dalam Kotak Hasil */
    .result-text { 
        font-size: 15px; 
        font-weight: bold; 
        margin: 0; 
    }
    
    /* Kontainer dan ukuran Icon Ribbon di atas */
    .logo-container { display: flex; justify-content: center; margin-top: 25px; margin-bottom: 15px; }
    .ribbon-icon { 
        background-color: #f8bbd0; 
        color: #e91e63; 
        font-size: 24px; 
        width: 50px; 
        height: 50px; 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        border-radius: 50%; 
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


st.markdown('<div class="logo-container"><div class="ribbon-icon">🎗️</div></div>', unsafe_allow_html=True)
st.markdown('<div class="welcome-text">Selamat datang! Mari kita langkah bersama untuk mengecek kesehatan payudara Anda<br>dengan penuh kepedulian dan kehangatan—selangkah demi selangkah.</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    html_image = get_image_download_link(image)
    st.markdown(html_image, unsafe_allow_html=True)
    
    img_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    
    if predicted_class == 1:
        hasil_teks = "Ganas"
        warna_box = "#ffebee"  
        warna_teks = "#c62828" 
    else:
        hasil_teks = "Jinak"
        warna_box = "#e0f2f1"  
        warna_teks = "#2e7d32" 
        
    st.markdown(
        f'<div class="result-box" style="background-color: {warna_box};"><p class="result-text" style="color: {warna_teks};">Hasil Prediksi : {hasil_teks}</p></div>', 
        unsafe_allow_html=True
    )

st.markdown('<div class="disclaimer-text">Mengingat ini adalah alat skrining dan edukasi, kami sangat menyarankan Anda untuk tetap berkonsultasi<br>dengan dokter spesialis demi mendapatkan diagnosis medis yang menyeluruh.</div>', unsafe_allow_html=True)
