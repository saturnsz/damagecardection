import streamlit as st
import cv2
import torch
import numpy as np
import base64
import os
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from models import load_cnn_model

# --- Page Config ---
st.set_page_config(
    page_title="Sistem Deteksi Kerusakan Mobil",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    /* Global Background & Overlay */
    [data-testid="stAppViewContainer"] {
        /* This will be overwritten dynamically by base64 string later, keeping this as fallback */
        background-color: #f8f9fa;
    }
    
    /* Make the top header transparent */
    [data-testid="stHeader"] {
        background: transparent !important;
    }
    
    /* Table Styling: Pertebal garis tabel */
    [data-testid="stTable"] table {
        border: 2px solid #ffffff !important;
    }
    [data-testid="stTable"] th, [data-testid="stTable"] td {
        border: 2px solid #ffffff !important;
        color: #ffffff !important;
    }
    [data-testid="stTable"] th {
        background-color: rgba(255, 255, 255, 0.15) !important;
        font-weight: 600 !important;
    }
    
    /* Add a border to the images */
    [data-testid="stImage"] img {
        border: 3px solid #ffffff;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .css-1d391kg {
        background-color: #ffffff;
        border-radius: 4px;
        padding: 2rem;
        border: 1px solid #eaeaea;
    }
    h1, h2, h3 {
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    .info-box {
        background-color: #f8f9fa;
        padding: 24px;
        border-radius: 16px; /* Ujung dibuat sangat tumpul */
        color: #333;
        font-weight: 400;
        margin-bottom: 20px;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15); /* Shadow lembut pengganti garis kaku */
    }
</style>
""", unsafe_allow_html=True)

# --- Models & Preprocessing ---
@st.cache_resource
def load_models():
    """Load both YOLO and CNN models."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load YOLOv8n for general Car validation (COCO dataset)
    yolo_val = YOLO("yolov8n.pt")
    
    # Load custom YOLOv9c
    yolo_model = YOLO("best (1).pt")
    
    # Load CNN ResNet+CBAM
    cnn_model = load_cnn_model("best_cnn_resnet_cbam.pth", device=device)
    
    return yolo_val, yolo_model, cnn_model, device

def preprocess_for_cnn(crop_img):
    """Preprocess the cropped image for ResNet18."""
    # Convert OpenCV BGR to RGB PIL Image
    image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    # Standard ResNet transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(pil_image).unsqueeze(0)
    return input_tensor

def set_background(image_path):
    """Set a background image using base64 encoding."""
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            
        # We add an overlay (background-blend-mode) to fade the image slightly, 
        # so it doesn't distract from the minimalist UI elements.
        st.markdown(
            f"""
            <style>
            [data-testid="stAppViewContainer"] {{
                background-image: linear-gradient(rgba(0, 0, 0, 0.50), rgba(0, 0, 0, 0.50)), url(data:image/jpeg;base64,{encoded_string});
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

# --- Labels Mapping ---
CNN_CLASSES = {0: "Ringan / Minor", 1: "Parah / Severe"}
YOLO_TARGET_CLASS = 1 # 'Rusak' is index 1

# --- Main App ---
def main():
    # Set the background first
    set_background("background.jpg")

    st.markdown('<h1 style="text-align: center; font-weight: 300; color: #ffffff;">Deteksi Kerusakan Bodi Mobil</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #dddddd; margin-bottom: 40px; font-weight: 600;">Hybrid YOLOv9c & CNN-ResNet-CBAM</h3>', unsafe_allow_html=True)

    with st.spinner("Memuat model AI... (Mohon tunggu)"):
        yolo_val, yolo_model, cnn_model, device = load_models()
    
    st.sidebar.markdown("### Panel Kontrol")
    page = st.sidebar.radio("Navigasi Halaman", ["Beranda", "Analisis Citra"])
    
    if page == "Beranda":
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Pintasan Beranda")
        st.sidebar.markdown("- [About / Informasi Sistem](#about)")
        st.sidebar.markdown("- [Contoh Data Kerusakan](#contoh-data)")
        st.sidebar.markdown("- [Support By](#support-by)")
        
        # --- Bagian About ---
        st.markdown('<div id="about"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            <h4>Selamat Datang di Sistem Analisis Otomatis!</h4>
            <p>Sistem ini menerapkan alur kerja hibrida sesuai rancangan Anda:</p>
            <ol>
                <li><b>Input:</b> Mengunggah citra mobil (tanpa database).</li>
                <li><b>Model 1 (YOLOv9c):</b> Deteksi dan lokalisasi area bodi.</li>
                <li><b>Proses (Crop Otomatis):</b> Memotong area rusak secara real-time.</li>
                <li><b>Model 2 (ResNet18+CBAM):</b> Klasifikasi tingkat keparahan geometri kerusakan.</li>
                <li><b>Output:</b> Visualisasi hasil analisis secara transparan.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # --- Bagian Contoh Data Klasifikasi ---
        st.markdown('<div id="contoh-data"></div>', unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<hr style="border-top: 1px solid rgba(255,255,255,0.2);">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #ffffff; margin-bottom: 30px;">Contoh Data Klasifikasi Deteksi Bodi Mobil</h3>', unsafe_allow_html=True)
        
        # Menggunakan rasio kolom [1, 2, 2, 1] agar gambar di tengah tidak terlalu besar
        col_space1, col_ex1, col_ex2, col_space2 = st.columns([1, 2, 2, 1])
        
        with col_ex1:
            st.markdown('<p style="text-align: center; color: #dddddd; font-size: 1.1rem; font-weight: 600;">Gambar 1: Bodi Normal</p>', unsafe_allow_html=True)
            if os.path.exists("gambar1.jpg"):
                st.image("gambar1.jpg", use_container_width=True)
            else:
                st.info("⚠️ Silakan simpan gambar mobil normal yang Anda kirim ke folder proyek dengan nama file: gambar1.jpg")
                
        with col_ex2:
            st.markdown('<p style="text-align: center; color: #dddddd; font-size: 1.1rem; font-weight: 600;">Gambar 2: Bodi Rusak</p>', unsafe_allow_html=True)
            if os.path.exists("gambar2.jpg"):
                st.image("gambar2.jpg", use_container_width=True)
            elif os.path.exists("gambar2.JPEG"):
                st.image("gambar2.JPEG", use_container_width=True)
            elif os.path.exists("gambar2.png"):
                st.image("gambar2.png", use_container_width=True)
            else:
                st.info("⚠️ Silakan simpan gambar mobil rusak yang Anda kirim ke folder proyek dengan nama file: gambar2.jpg")

        # --- Bagian Support By (Footer) ---
        st.markdown('<div id="support-by"></div>', unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<hr style="border-top: 1px solid rgba(255,255,255,0.2);">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #ffffff; margin-bottom: 20px;">Support By</h3>', unsafe_allow_html=True)
        
        def get_base64_of_bin_file(bin_file):
            if os.path.exists(bin_file):
                with open(bin_file, 'rb') as f:
                    data = f.read()
                return base64.b64encode(data).decode()
            return ""

        logo1_b64 = get_base64_of_bin_file("logo 1.jpg")
        logo2_b64 = get_base64_of_bin_file("logo 2.jpg")
        logo3_b64 = get_base64_of_bin_file("logo 3.png")

        img1_tag = f'<img src="data:image/jpeg;base64,{logo1_b64}" style="height: 120px; width: 100%; object-fit: contain; margin-bottom: 15px;" alt="tukul speed"/>' if logo1_b64 else ''
        img2_tag = f'<img src="data:image/jpeg;base64,{logo2_b64}" style="height: 120px; width: 100%; object-fit: contain; margin-bottom: 15px;" alt="devv garage"/>' if logo2_b64 else ''
        img3_tag = f'<img src="data:image/png;base64,{logo3_b64}" style="height: 120px; width: 100%; object-fit: contain; margin-bottom: 15px;" alt="new atlantis"/>' if logo3_b64 else ''

        footer_html = f"""
        <div style="display: flex; flex-wrap: wrap; justify-content: center; align-items: center; gap: 40px; margin-bottom: 30px; background-color: rgba(255,255,255,0.05); padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
            <div style="text-align: center; color: #ffffff; font-weight: 600; width: 180px;">
                {img1_tag}
                <div style="font-size: 1.1rem;">Tukul Speed</div>
            </div>
            <div style="text-align: center; color: #ffffff; font-weight: 600; width: 180px;">
                {img2_tag}
                <div style="font-size: 1.1rem;">Devv Garage</div>
            </div>
            <div style="text-align: center; color: #ffffff; font-weight: 600; width: 220px;">
                {img3_tag}
                <div style="font-size: 1.1rem; line-height: 1.2;">New Atlantis Body Repair<br>& Paint Specialist</div>
            </div>
            <div style="text-align: center; color: #ffffff; font-weight: 800; font-size: 2rem; margin-left: 20px; letter-spacing: 1px;">
                #daridoaibu
            </div>
        </div>
        """
        st.markdown(footer_html, unsafe_allow_html=True)

    elif page == "Analisis Citra":
        st.sidebar.markdown("---")
        st.sidebar.markdown("Unggah gambar mobil untuk menganalisis lokasi dan tingkat kerusakan bodi secara otomatis.")
        
        uploaded_file = st.sidebar.file_uploader("Upload Citra Mobil", type=["jpg", "jpeg", "png"])
        confidence_threshold = st.sidebar.slider("YOLO Confidence Threshold", min_value=0.1, max_value=1.0, value=0.25, step=0.05)

        if uploaded_file is not None:
            # Load image
            image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(image_bytes, 1) # BGR image for processing
            img_display = img.copy()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Citra Input")
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                st.markdown("### Hasil Analisis")
                
                with st.spinner("Melakukan Validasi Objek..."):
                    # Pre-validation using COCO model
                    val_results = yolo_val.predict(img, conf=0.15)
                    # COCO classes for vehicles: 2 (car), 3 (motorcycle), 5 (bus), 7 (truck)
                    vehicle_classes = [2, 3, 5, 7]
                    
                    is_vehicle = False
                    for box in val_results[0].boxes:
                        if int(box.cls[0].item()) in vehicle_classes:
                            is_vehicle = True
                            break
                
                if not is_vehicle:
                    st.error("Validasi Gagal: Sistem tidak mendeteksi adanya mobil pada gambar ini. Mohon unggah citra kendaraan yang benar.")
                else:
                    with st.spinner("Menjalankan Deteksi YOLOv9c..."):
                        # Run Custom YOLO
                        results = yolo_model.predict(img, conf=confidence_threshold)
                
                    detections = results[0].boxes
                    
                    if len(detections) == 0:
                        st.error("YOLOv9c tidak mendeteksi bodi/kerusakan pada gambar ini dengan confidence tersebut. Coba turunkan slider.")
                    else:
                        st.success(f"Ditemukan {len(detections)} deteksi area pada mobil.")
                        
                        analysis_results = []
                        
                        # Process each detection
                        for i, box in enumerate(detections):
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            
                            # CROP OTOMATIS AREA DETEKSI
                            # Ensure coordinates are within image bounds
                            h, w, _ = img.shape
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            cropped_img = img[y1:y2, x1:x2]
                            
                            if cropped_img.size == 0:
                                continue
                                
                            # Run CNN + ResNet-CBAM Classification for ALL detections
                            input_tensor = preprocess_for_cnn(cropped_img).to(device)
                            with torch.no_grad():
                                outputs = cnn_model(input_tensor)
                                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                                pred_class_id = torch.argmax(probabilities, dim=1).item()
                                pred_confidence = probabilities[0][pred_class_id].item()
                                
                            severity_label = CNN_CLASSES.get(pred_class_id, "Unknown")
                            yolo_label_name = "Rusak" if cls_id == 1 else "Normal"
                            
                            # Store result
                            analysis_results.append({
                                "ID": f"Deteksi #{i+1}",
                                "YOLO Deteksi": yolo_label_name,
                                "CNN Keparahan": severity_label,
                                "CNN Confidence": f"{pred_confidence*100:.1f}%",
                                "YOLO Confidence": f"{conf*100:.1f}%"
                            })
                            
                            # Draw Bounding Box & Label on Image
                            # Color based on CNN prediction: Green for Class 0 (Ringan/Normal), Red for Class 1 (Parah)
                            color = (0, 0, 255) if pred_class_id == 1 else (0, 255, 0)
                            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 3)
                            
                            label_text = f"#{i+1}: {severity_label}"
                            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(img_display, (x1, y1-th-10), (x1+tw+10, y1), color, -1)
                            cv2.putText(img_display, label_text, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
                        st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), use_container_width=True, caption="Citra dengan Deteksi Kerusakan & Klasifikasi")
                        
                        if len(analysis_results) > 0:
                            st.markdown("#### Rincian Keparahan Geometri (ResNet-CBAM)")
                            st.table(analysis_results)
                        else:
                            st.info("Area terdeteksi berstatus 'Normal' (tidak dianalisis keparahannya).")
        else:
            st.info("Silakan unggah citra mobil melalui panel di sebelah kiri untuk memulai analisis.")

if __name__ == "__main__":
    main()
