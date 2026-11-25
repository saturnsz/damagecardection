import streamlit as st
import cv2
import torch
import numpy as np
import base64
import os
import tempfile
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from models import load_cnn_model
from preprocess.quality_gate import run_gate
from triage.rules import triage, get_disclaimer
from report.pdf_builder import build_report, generate_report_id, cleanup_temp_files

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
        border-radius: 16px;
        color: #333;
        font-weight: 400;
        margin-bottom: 20px;
        font-size: 0.95rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    /* Triage Card Styling */
    .triage-card {
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        color: #fff;
        font-weight: 500;
    }
    .triage-minor { background: linear-gradient(135deg, #28a745, #20c997); }
    .triage-moderate { background: linear-gradient(135deg, #ffc107, #fd7e14); color: #333; }
    .triage-severe { background: linear-gradient(135deg, #dc3545, #c82333); }
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
    image_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
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

# --- Helper: Save image to temp file for PDF ---
def _save_temp_image(image_bgr, prefix="img"):
    """Save a BGR image to a temporary JPEG file. Returns the path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix=prefix, delete=False)
    # Compress to keep PDF small (PD-5)
    cv2.imwrite(tmp.name, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    tmp.close()
    return tmp.name

# --- Helper: Compute DSI from detections ---
def compute_dsi(analysis_results):
    """Compute Damage Severity Index as a percentage.
    
    DSI = (number of 'Parah / Severe' detections / total detections) * 100
    combined with average CNN confidence for severe detections.
    """
    if not analysis_results:
        return 0.0
    
    total = len(analysis_results)
    severe_count = 0
    severe_conf_sum = 0.0
    
    for det in analysis_results:
        if "Parah" in det.get("CNN Keparahan", "") or "Severe" in det.get("CNN Keparahan", ""):
            severe_count += 1
            conf_str = det.get("CNN Confidence", "0%").replace("%", "")
            try:
                severe_conf_sum += float(conf_str)
            except ValueError:
                pass
    
    if severe_count == 0:
        # All minor — DSI is low, proportional to damage density
        return min(14.9, (total / 10.0) * 14.9)
    
    # Weighted DSI: ratio of severe * average confidence of severe detections
    severity_ratio = severe_count / total
    avg_conf = severe_conf_sum / severe_count if severe_count > 0 else 50.0
    
    # Scale: severity_ratio * avg_conf gives 0-100 range
    dsi = severity_ratio * avg_conf
    return min(100.0, max(0.0, dsi))

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
                <li><b>Quality Gate:</b> Validasi kualitas gambar (blur & kecerahan) sebelum analisis.</li>
                <li><b>Model 1 (YOLOv9c):</b> Deteksi dan lokalisasi area bodi.</li>
                <li><b>Proses (Crop Otomatis):</b> Memotong area rusak secara real-time.</li>
                <li><b>Model 2 (ResNet18+CBAM):</b> Klasifikasi tingkat keparahan geometri kerusakan.</li>
                <li><b>Triase & SOP:</b> Rekomendasi jalur penanganan dan SOP bengkel.</li>
                <li><b>Laporan PDF:</b> Unduh laporan inspeksi awal untuk dokumentasi.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

        # --- Bagian Contoh Data Klasifikasi ---
        st.markdown('<div id="contoh-data"></div>', unsafe_allow_html=True)
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<hr style="border-top: 1px solid rgba(255,255,255,0.2);">', unsafe_allow_html=True)
        st.markdown('<h3 style="text-align: center; color: #ffffff; margin-bottom: 30px;">Contoh Data Klasifikasi Deteksi Bodi Mobil</h3>', unsafe_allow_html=True)
        
        col_space1, col_ex1, col_ex2, col_space2 = st.columns([1, 2, 2, 1])
        
        with col_ex1:
            st.markdown('<p style="text-align: center; color: #dddddd; font-size: 1.1rem; font-weight: 600;">Gambar 1: Bodi Normal</p>', unsafe_allow_html=True)
            if os.path.exists("gambar1.jpg"):
                st.image("gambar1.jpg", use_container_width=True)
            else:
                st.info("Silakan simpan gambar mobil normal ke folder proyek: gambar1.jpg")
                
        with col_ex2:
            st.markdown('<p style="text-align: center; color: #dddddd; font-size: 1.1rem; font-weight: 600;">Gambar 2: Bodi Rusak</p>', unsafe_allow_html=True)
            if os.path.exists("gambar2.jpg"):
                st.image("gambar2.jpg", use_container_width=True)
            elif os.path.exists("gambar2.JPEG"):
                st.image("gambar2.JPEG", use_container_width=True)
            elif os.path.exists("gambar2.png"):
                st.image("gambar2.png", use_container_width=True)
            else:
                st.info("Silakan simpan gambar mobil rusak ke folder proyek: gambar2.jpg")

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
                <div style="font-size: 1.1rem; line-height: 1.2;">New Atlantis Body Repair<br>&amp; Paint Specialist</div>
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

            # ===================================================================
            # STEP 1: IMAGE QUALITY GATEKEEPER
            # ===================================================================
            with st.spinner("Memeriksa kualitas gambar..."):
                gate_result = run_gate(img)
            
            gate_status = gate_result["status"]
            gate_message = gate_result["message"]
            
            # Display gate result
            if gate_status == "rejected_blur":
                st.error(f"Gambar Ditolak: Gambar terlalu buram untuk dianalisis. "
                         f"(Skor blur: {gate_result['blur_score']:.1f})")
                st.warning("Mohon unggah ulang foto yang lebih tajam dan jelas.")
                st.info(f"Detail: {gate_message}")
                return  # Stop processing
            
            elif gate_status == "enhanced":
                st.info(f"Peningkatan Kualitas: Gambar gelap terdeteksi dan telah diperbaiki otomatis dengan CLAHE. "
                        f"(Kecerahan: {gate_result['brightness_score']:.1f})")
            
            elif gate_status == "ok":
                st.success(f"Kualitas gambar baik. (Blur: {gate_result['blur_score']:.1f}, "
                           f"Kecerahan: {gate_result['brightness_score']:.1f})")
            
            # Use the (possibly enhanced) image for further processing
            img_processed = gate_result["image"]
            img_original = gate_result["original_image"]
            img_display = img_processed.copy()

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("### Citra Input")
                if gate_status == "enhanced":
                    tab_orig, tab_enhanced = st.tabs(["Gambar Asli", "Setelah CLAHE"])
                    with tab_orig:
                        st.image(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB), use_container_width=True)
                    with tab_enhanced:
                        st.image(cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB), use_container_width=True)
                else:
                    st.image(cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col2:
                st.markdown("### Hasil Analisis")
                
                with st.spinner("Melakukan Validasi Objek..."):
                    # Pre-validation using COCO model
                    val_results = yolo_val.predict(img_processed, conf=0.15)
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
                        results = yolo_model.predict(img_processed, conf=confidence_threshold)
                
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
                            
                            h, w, _ = img_processed.shape
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            cropped_img = img_processed[y1:y2, x1:x2]
                            
                            if cropped_img.size == 0:
                                continue
                                
                            input_tensor = preprocess_for_cnn(cropped_img).to(device)
                            with torch.no_grad():
                                outputs = cnn_model(input_tensor)
                                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                                pred_class_id = torch.argmax(probabilities, dim=1).item()
                                pred_confidence = probabilities[0][pred_class_id].item()
                                
                            severity_label = CNN_CLASSES.get(pred_class_id, "Unknown")
                            yolo_label_name = "Rusak" if cls_id == 1 else "Normal"
                            
                            analysis_results.append({
                                "ID": f"Deteksi #{i+1}",
                                "YOLO Deteksi": yolo_label_name,
                                "CNN Keparahan": severity_label,
                                "CNN Confidence": f"{pred_confidence*100:.1f}%",
                                "YOLO Confidence": f"{conf*100:.1f}%"
                            })
                            
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
                            
                            # =======================================================
                            # STEP 2: TRIASE KLAIM & REKOMENDASI SOP
                            # =======================================================
                            st.markdown("---")
                            st.markdown("### Rekomendasi Triase & SOP Bengkel")
                            
                            # Compute DSI
                            dsi_value = compute_dsi(analysis_results)
                            
                            # Run triage
                            triage_result = triage(
                                dsi=dsi_value,
                                num_boxes=len(analysis_results),
                            )
                            
                            # Display DSI prominently
                            dsi_color = triage_result["color"]
                            st.markdown(f"""
                            <div style="text-align: center; padding: 15px; border-radius: 10px; 
                                        background: linear-gradient(135deg, {dsi_color}22, {dsi_color}44);
                                        border: 2px solid {dsi_color}; margin-bottom: 15px;">
                                <h2 style="color: {dsi_color}; margin: 0;">Damage Severity Index (DSI)</h2>
                                <h1 style="color: {dsi_color}; margin: 5px 0; font-size: 3rem;">{dsi_value:.1f}%</h1>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Triage Card (TR-8)
                            level_lower = triage_result["level"].lower()
                            card_class = f"triage-{level_lower}"
                            
                            st.markdown(f"""
                            <div class="triage-card {card_class}">
                                <h3 style="margin-top: 0;">{triage_result["icon"]} Tingkat: {triage_result["level"]} ({triage_result["level_id"]})</h3>
                                <p><b>Rekomendasi Jalur:</b> {triage_result["path"]}</p>
                                <p><b>SOP yang Direkomendasikan:</b> {triage_result["sop"]}</p>
                                <p><b>Estimasi Waktu:</b> {triage_result["estimate"]}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Override notice (TR-7)
                            if triage_result["override"]:
                                st.warning(f"Override Diterapkan: {triage_result['override_reason']}")
                            
                            # Disclaimer (TR-5)
                            disclaimer_text = get_disclaimer()
                            st.caption(f"*{disclaimer_text}*")
                            
                            # =======================================================
                            # STEP 3: LAPORAN INSPEKSI PDF
                            # =======================================================
                            st.markdown("---")
                            st.markdown("### Unduh Laporan Inspeksi")
                            
                            # Save images to temp files for PDF (PD-5)
                            temp_files = []
                            try:
                                original_tmp = _save_temp_image(img_original, prefix="orig_")
                                temp_files.append(original_tmp)
                                
                                annotated_tmp = _save_temp_image(img_display, prefix="annot_")
                                temp_files.append(annotated_tmp)
                                
                                enhanced_tmp = None
                                if gate_status == "enhanced":
                                    enhanced_tmp = _save_temp_image(img_processed, prefix="enh_")
                                    temp_files.append(enhanced_tmp)
                                
                                # Build PDF context
                                report_id = generate_report_id()
                                from datetime import datetime
                                timestamp_str = datetime.now().strftime("%d %B %Y, %H:%M:%S WIB")
                                
                                pdf_context = {
                                    "report_id": report_id,
                                    "timestamp": timestamp_str,
                                    "original_image_path": original_tmp,
                                    "annotated_image_path": annotated_tmp,
                                    "enhanced_image_path": enhanced_tmp,
                                    "gate_status": gate_status,
                                    "gate_message": gate_message,
                                    "detections": analysis_results,
                                    "dsi": dsi_value,
                                    "triage_result": triage_result,
                                    "disclaimer": disclaimer_text,
                                }
                                
                                # Build the PDF (PD-8)
                                pdf_bytes = build_report(pdf_context)
                                
                                # Download button
                                st.download_button(
                                    label="Unduh Laporan Inspeksi (PDF)",
                                    data=pdf_bytes,
                                    file_name=f"laporan_inspeksi_{report_id}.pdf",
                                    mime="application/pdf",
                                    type="primary",
                                )
                                
                                st.caption(f"ID Laporan: `{report_id}`")
                                
                            except Exception as e:
                                st.error(f"Gagal membuat laporan PDF: {str(e)}")
                            finally:
                                # Cleanup temp files (PD-9)
                                cleanup_temp_files(temp_files)
                        else:
                            st.info("Area terdeteksi berstatus 'Normal' (tidak dianalisis keparahannya).")
        else:
            st.info("Silakan unggah citra mobil melalui panel di sebelah kiri untuk memulai analisis.")

if __name__ == "__main__":
    main()
