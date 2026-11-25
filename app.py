import streamlit as st
import cv2
import torch
import numpy as np
import base64
import hashlib
import os
import tempfile
from datetime import datetime
from PIL import Image
from ultralytics import YOLO
from torchvision import transforms
from models import load_cnn_model
from preprocess.quality_gate import run_gate
from severity.clustering import cluster_boxes
from severity.dsi import compute_dsi, select_body_box
from severity.rules import classify
from triage.mapping import triage, get_disclaimer
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
    /* Severity Multi-Factor Card Styling */
    .severity-card {
        padding: 18px 20px;
        border-radius: 12px;
        margin: 12px 0 16px 0;
        background: rgba(15, 23, 42, 0.78);
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(0,0,0,0.28);
    }
    .severity-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        margin-bottom: 8px;
    }
    .severity-reason {
        font-size: 0.92rem;
        color: #e2e8f0;
        margin: 4px 0 14px 0;
        line-height: 1.45;
    }
    .severity-metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
    }
    .severity-metric-box {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.16);
        border-radius: 8px;
        padding: 10px 8px;
        text-align: center;
    }
    .severity-metric-label {
        font-size: 0.75rem;
        color: #cbd5e1;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .severity-metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #ffffff;
        margin-top: 2px;
    }
    /* Triage Card Styling */
    .triage-card {
        padding: 20px;
        border-radius: 12px;
        margin: 10px 0;
        color: #fff;
        font-weight: 500;
    }
    .triage-card h3 { color: inherit; }
    .triage-none { background: linear-gradient(135deg, #17a2b8, #138496); }
    .triage-minor { background: linear-gradient(135deg, #28a745, #20c997); }
    .triage-moderate { background: linear-gradient(135deg, #ffc107, #fd7e14); color: #212529; }
    .triage-severe { background: linear-gradient(135deg, #dc3545, #c82333); }
    /* Login Card Styling */
    [data-testid="stForm"] {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.22);
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    [data-testid="stForm"] label,
    [data-testid="stForm"] p {
        color: #f8fafc !important;
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
def _save_temp_image(image_bgr, prefix="img", max_dim=1600):
    """Downscale if oversized and save BGR image to temp JPEG for compact PDF (PD-5)."""
    h, w = image_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image_bgr = cv2.resize(
            image_bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", prefix=prefix, delete=False)
    cv2.imwrite(tmp.name, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    tmp.close()
    return tmp.name

def _config_signature() -> str:
    """Return mtime fingerprint of YAML configs so edits take effect immediately."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    parts = []
    for name in ("quality_gate.yaml", "severity.yaml", "triage.yaml"):
        path = os.path.join(base_dir, name)
        if os.path.isfile(path):
            parts.append(f"{name}:{int(os.path.getmtime(path))}")
    return "|".join(parts)


# --- Labels Mapping & Demo Auth Credentials ---
CNN_CLASSES = {0: "Ringan / Minor", 1: "Rusak"}
YOLO_TARGET_CLASS = 1 # 'Rusak' is index 1
DEMO_USERS = {"user": "user"}


def verify_credentials(username: str, password: str) -> bool:
    """Validate login credentials against temporary non-DB credential map."""
    clean_user = (username or "").strip()
    expected = DEMO_USERS.get(clean_user)
    return expected is not None and password == expected


def render_login_page() -> None:
    """Render login form with note for direct login while database is not connected."""
    _, col_center, _ = st.columns([1, 1.3, 1])
    with col_center:
        st.markdown(
            '<h3 style="text-align: center; color: #ffffff; margin-bottom: 12px;">Login Sistem</h3>',
            unsafe_allow_html=True,
        )
        st.info(
            "**Catatan:** Sistem saat ini belum terkoneksi ke database. "
            "Anda dapat langsung login menggunakan akun berikut:\n"
            "- **Username:** `user`\n"
            "- **Password:** `user`"
        )

        with st.form("login_form", clear_on_submit=False):
            username_input = st.text_input("Username", placeholder="Masukkan username (user)")
            password_input = st.text_input("Password", type="password", placeholder="Masukkan password (user)")
            submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

            if submitted:
                if verify_credentials(username_input, password_input):
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username_input.strip()
                    st.rerun()
                else:
                    st.error("Username atau password salah. Gunakan username `user` dan password `user`.")


# --- Main App ---
def main():
    # Set the background first
    set_background("background.jpg")

    st.markdown('<h1 style="text-align: center; font-weight: 300; color: #ffffff;">Deteksi Kerusakan Bodi Mobil</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #dddddd; margin-bottom: 40px; font-weight: 600;">Hybrid YOLOv9c & CNN-ResNet-CBAM</h3>', unsafe_allow_html=True)

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        render_login_page()
        return

    with st.spinner("Memuat model AI... (Mohon tunggu)"):
        yolo_val, yolo_model, cnn_model, device = load_models()
    
    st.sidebar.markdown("### Panel Kontrol")
    st.sidebar.success(f"Login sebagai: **{st.session_state.get('username', 'user')}**")
    if st.sidebar.button("Logout", key="logout_button", use_container_width=True):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.session_state.pop("analysis_key", None)
        st.session_state.pop("analysis_cache", None)
        st.rerun()

    st.sidebar.markdown("---")
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
                <li><b>Model 2 (ResNet18+CBAM):</b> Validasi status kerusakan area deteksi.</li>
                <li><b>Penilaian Keparahan (DSI):</b> Perhitungan luas gabungan kerusakan, pengelompokan klaster titik, dan penentuan label <i>Ringan / Sedang / Parah</i>.</li>
                <li><b>Triase &amp; SOP:</b> Pemetaan label keparahan ke rekomendasi jalur penanganan dan SOP bengkel.</li>
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
            raw_bytes = uploaded_file.getvalue()
            analysis_key = f"{hashlib.md5(raw_bytes).hexdigest()}_{confidence_threshold:.2f}_{_config_signature()}"

            if st.session_state.get("analysis_key") != analysis_key:
                image_bytes = np.asarray(bytearray(raw_bytes), dtype=np.uint8)
                img = cv2.imdecode(image_bytes, 1)  # BGR image for processing

                if img is None:
                    st.error("File gambar tidak dapat dibaca atau rusak. Mohon unggah file JPG/JPEG/PNG yang valid.")
                    return

                # STEP 1: IMAGE QUALITY GATEKEEPER
                with st.spinner("Memeriksa kualitas gambar..."):
                    gate_result = run_gate(img)

                cache = {
                    "gate_result": gate_result,
                    "is_vehicle": None,
                    "num_raw_detections": 0,
                    "img_display": None,
                    "analysis_results": [],
                    "dsi_result": None,
                    "dsi_value": 0.0,
                    "clusters": [],
                    "severity_result": None,
                    "triage_result": None,
                    "disclaimer_text": "",
                    "report_id": None,
                    "pdf_bytes": None,
                    "pdf_error": None,
                }

                if gate_result["status"] != "rejected_blur":
                    img_processed = gate_result["image"]
                    img_original = gate_result["original_image"]
                    img_display = img_processed.copy()

                    with st.spinner("Melakukan Validasi Objek..."):
                        val_results = yolo_val.predict(img_processed, conf=0.15)
                        vehicle_classes = [2, 3, 5, 7]
                        vehicle_boxes = [
                            tuple(map(int, box.xyxy[0].tolist()))
                            for box in val_results[0].boxes
                            if int(box.cls[0].item()) in vehicle_classes
                        ]
                        is_vehicle = len(vehicle_boxes) > 0

                    cache["is_vehicle"] = is_vehicle

                    if is_vehicle:
                        with st.spinner("Menjalankan Deteksi YOLOv9c..."):
                            results = yolo_model.predict(img_processed, conf=confidence_threshold)

                        detections = results[0].boxes
                        cache["num_raw_detections"] = len(detections)

                        raw_items = []
                        damage_boxes = []
                        h, w, _ = img_processed.shape

                        for i, box in enumerate(detections):
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
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

                            cnn_label = CNN_CLASSES.get(pred_class_id, "Unknown")
                            yolo_label_name = "Rusak" if cls_id == YOLO_TARGET_CLASS else "Normal"

                            dmg_idx = None
                            if cls_id == YOLO_TARGET_CLASS:
                                dmg_idx = len(damage_boxes)
                                damage_boxes.append((x1, y1, x2, y2))

                            raw_items.append({
                                "det_index": i + 1,
                                "cls_id": cls_id,
                                "conf": conf,
                                "bbox": (x1, y1, x2, y2),
                                "pred_class_id": pred_class_id,
                                "pred_confidence": pred_confidence,
                                "cnn_label": cnn_label,
                                "yolo_label": yolo_label_name,
                                "dmg_idx": dmg_idx,
                            })

                        # STEP 2: MULTI-FACTOR SEVERITY ASSESSMENT (DSI + CLUSTERING + RULES)
                        body_box = select_body_box(
                            vehicle_boxes,
                            damage_boxes=damage_boxes,
                            shape=img_processed.shape,
                        )
                        dsi_result = compute_dsi(
                            damage_boxes,
                            shape=img_processed.shape,
                            body_box=body_box,
                        )
                        clusters = cluster_boxes(
                            damage_boxes,
                            shape=img_processed.shape,
                            body_area=dsi_result["body_area"],
                        )
                        max_cluster_ratio = max((c["ratio"] for c in clusters), default=0.0)
                        severity_result = classify(
                            dsi=dsi_result["dsi"],
                            n_clusters=len(clusters),
                            max_cluster_ratio=max_cluster_ratio,
                        )

                        # STEP 3: TRIASE & SOP MAPPING FROM SEVERITY LABEL
                        triage_result = triage(severity_result["label"])
                        disclaimer_text = get_disclaimer()

                        # Map each damage box index to its cluster ID (UI-3)
                        box_to_cluster = {}
                        for cluster in clusters:
                            for dmg_idx in cluster["indices"]:
                                box_to_cluster[dmg_idx] = cluster["cluster_id"]

                        # Draw cluster grouping outlines when a cluster merges >1 box
                        for cluster in clusters:
                            if len(cluster["boxes"]) > 1:
                                cx1, cy1, cx2, cy2 = cluster["bbox"]
                                cv2.rectangle(img_display, (cx1, cy1), (cx2, cy2), (0, 215, 255), 2)

                        analysis_results = []
                        for item in raw_items:
                            x1, y1, x2, y2 = item["bbox"]
                            det_num = item["det_index"]
                            cls_id = item["cls_id"]
                            pred_class_id = item["pred_class_id"]
                            cnn_label = item["cnn_label"]

                            if cls_id == YOLO_TARGET_CLASS:
                                cid = box_to_cluster.get(item["dmg_idx"], 1)
                                cluster_str = f"Titik {cid}"
                                color = (0, 0, 255) if pred_class_id == 1 else (0, 140, 255)
                                label_text = f"Titik {cid} | #{det_num}: {cnn_label}"
                            else:
                                cluster_str = "-"
                                color = (0, 200, 0)
                                label_text = f"#{det_num}: Normal"

                            analysis_results.append({
                                "ID": f"Deteksi #{det_num}",
                                "Klaster": cluster_str,
                                "YOLO Deteksi": item["yolo_label"],
                                "CNN Keparahan": cnn_label,
                                "CNN Confidence": f"{item['pred_confidence']*100:.1f}%",
                                "YOLO Confidence": f"{item['conf']*100:.1f}%",
                            })

                            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 3)
                            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                            if y1 - th - 10 >= 0:
                                lbl_y1, lbl_y2 = y1 - th - 10, y1
                                txt_y = y1 - 5
                            else:
                                lbl_y1, lbl_y2 = y1, min(h, y1 + th + 10)
                                txt_y = min(h - 3, y1 + th + 5)
                            cv2.rectangle(img_display, (x1, lbl_y1), (min(w, x1 + tw + 10), lbl_y2), color, -1)
                            cv2.putText(img_display, label_text, (x1 + 5, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                        cache["img_display"] = img_display
                        cache["analysis_results"] = analysis_results
                        cache["dsi_result"] = dsi_result
                        cache["dsi_value"] = dsi_result["dsi"]
                        cache["clusters"] = clusters
                        cache["severity_result"] = severity_result
                        cache["triage_result"] = triage_result
                        cache["disclaimer_text"] = disclaimer_text

                        # STEP 4: BUILD PDF ONCE AND CACHE BYTES IN SESSION STATE (UI-4)
                        temp_files = []
                        try:
                            original_tmp = _save_temp_image(img_original, prefix="orig_")
                            temp_files.append(original_tmp)

                            annotated_tmp = _save_temp_image(img_display, prefix="annot_")
                            temp_files.append(annotated_tmp)

                            enhanced_tmp = None
                            if gate_result["status"] == "enhanced":
                                enhanced_tmp = _save_temp_image(img_processed, prefix="enh_")
                                temp_files.append(enhanced_tmp)

                            report_id = generate_report_id()
                            timestamp_str = datetime.now().strftime("%d %B %Y, %H:%M:%S WIB")

                            pdf_context = {
                                "report_id": report_id,
                                "timestamp": timestamp_str,
                                "original_image_path": original_tmp,
                                "annotated_image_path": annotated_tmp,
                                "enhanced_image_path": enhanced_tmp,
                                "gate_status": gate_result["status"],
                                "gate_message": gate_result["message"],
                                "detections": analysis_results,
                                "dsi": dsi_result["dsi"],
                                "damage_area": dsi_result["damage_area"],
                                "body_area": dsi_result["body_area"],
                                "used_fallback_area": dsi_result["used_fallback"],
                                "n_clusters": len(clusters),
                                "max_cluster_ratio": max_cluster_ratio,
                                "severity_label": severity_result["label"],
                                "severity_reason": severity_result["reason"],
                                "triage_result": triage_result,
                                "disclaimer": disclaimer_text,
                            }
                            cache["report_id"] = report_id
                            cache["pdf_bytes"] = build_report(pdf_context)
                        except Exception as e:
                            cache["pdf_error"] = str(e)
                        finally:
                            cleanup_temp_files(temp_files)

                st.session_state["analysis_key"] = analysis_key
                st.session_state["analysis_cache"] = cache

            cache = st.session_state["analysis_cache"]
            gate_result = cache["gate_result"]
            gate_status = gate_result["status"]
            gate_message = gate_result["message"]

            if gate_status == "rejected_blur":
                st.error(
                    f"Gambar Ditolak: Gambar terlalu buram untuk dianalisis. "
                    f"(Skor blur: {gate_result['blur_score']:.1f})"
                )
                st.warning("Mohon unggah ulang foto yang lebih tajam dan jelas.")
                st.info(f"Detail: {gate_message}")
                return
            elif gate_status == "enhanced":
                st.info(
                    f"Peningkatan Kualitas: Gambar gelap terdeteksi dan telah diperbaiki otomatis dengan CLAHE. "
                    f"(Kecerahan: {gate_result['brightness_score']:.1f})"
                )
            elif gate_status == "ok":
                st.success(
                    f"Kualitas gambar baik. (Blur: {gate_result['blur_score']:.1f}, "
                    f"Kecerahan: {gate_result['brightness_score']:.1f})"
                )

            img_processed = gate_result["image"]
            img_original = gate_result["original_image"]

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

                if not cache["is_vehicle"]:
                    st.error("Validasi Gagal: Sistem tidak mendeteksi adanya mobil pada gambar ini. Mohon unggah citra kendaraan yang benar.")
                else:
                    if cache["num_raw_detections"] == 0:
                        st.info("YOLOv9c tidak mendeteksi kerusakan pada gambar ini pada confidence tersebut (coba turunkan slider jika memang terdapat kerusakan ringan).")
                    else:
                        st.success(f"Ditemukan {cache['num_raw_detections']} deteksi area pada mobil.")

                    img_display = cache["img_display"]
                    analysis_results = cache["analysis_results"]
                    dsi_result = cache["dsi_result"]
                    severity_result = cache["severity_result"]
                    triage_result = cache["triage_result"]
                    disclaimer_text = cache["disclaimer_text"]

                    if img_display is not None:
                        st.image(
                            cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB),
                            use_container_width=True,
                            caption="Citra Hasil Deteksi Kerusakan & Penomoran Titik Klaster",
                        )

                    # STEP 2: PENILAIAN KEPARAHAN MULTI-FAKTOR & DSI (UI-1, UI-2)
                    if severity_result and dsi_result:
                        st.markdown("### Penilaian Keparahan & DSI")
                        sev_color = severity_result["color"]
                        badge_fg = "#212529" if severity_result["level"] == "Moderate" else "#ffffff"
                        area_src_text = (
                            "Cadangan luas gambar (bodi tidak terdeteksi penuh)"
                            if dsi_result["used_fallback"]
                            else "Bounding box bodi kendaraan"
                        )

                        st.markdown(f"""
                        <div class="severity-card" style="border: 2px solid {sev_color};">
                            <span class="severity-badge" style="background-color: {sev_color}; color: {badge_fg};">
                                {severity_result["icon"]} {severity_result["label"]}
                            </span>
                            <div class="severity-reason">
                                <b>Alasan:</b> {severity_result["reason"]}
                            </div>
                            <div class="severity-metrics">
                                <div class="severity-metric-box">
                                    <div class="severity-metric-label">DSI</div>
                                    <div class="severity-metric-value" style="color: {sev_color};">{severity_result["dsi"]:.1f}%</div>
                                </div>
                                <div class="severity-metric-box">
                                    <div class="severity-metric-label">Titik Kerusakan</div>
                                    <div class="severity-metric-value">{severity_result["n_clusters"]}</div>
                                </div>
                                <div class="severity-metric-box">
                                    <div class="severity-metric-label">Klaster Dominan</div>
                                    <div class="severity-metric-value">{severity_result["max_cluster_pct"]:.1f}%</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.caption(
                            f"Luas gabungan kerusakan (union): {dsi_result['damage_area']:,} px | "
                            f"Luas acuan bodi: {dsi_result['body_area']:,} px ({area_src_text})"
                        )

                    if analysis_results:
                        st.markdown("#### Rincian Deteksi Area (YOLOv9c & ResNet-CBAM)")
                        st.table(analysis_results)

                    # STEP 3: TRIASE KLAIM & REKOMENDASI SOP (TR-1..TR-6)
                    if triage_result and severity_result:
                        st.markdown("---")
                        st.markdown("### Rekomendasi Triase & SOP Bengkel")

                        level_lower = triage_result["level"].lower()
                        card_class = f"triage-{level_lower}"

                        st.markdown(f"""
                        <div class="triage-card {card_class}">
                            <h3 style="margin-top: 0;">{triage_result["icon"]} {severity_result["label"]}</h3>
                            <p><b>Rekomendasi Jalur:</b> {triage_result["path"]}</p>
                            <p><b>SOP yang Direkomendasikan:</b> {triage_result["sop"]}</p>
                            <p><b>Estimasi Waktu:</b> {triage_result["estimate"]}</p>
                            <p style="margin-bottom: 0;"><b>Dasar Penilaian:</b> {severity_result["reason"]}</p>
                        </div>
                        """, unsafe_allow_html=True)

                        st.caption(f"*{disclaimer_text}*")

                    # STEP 4: LAPORAN INSPEKSI PDF (PD-1..PD-10)
                    st.markdown("---")
                    st.markdown("### Unduh Laporan Inspeksi")

                    if cache["pdf_error"]:
                        st.error(f"Gagal membuat laporan PDF: {cache['pdf_error']}")
                    elif cache["pdf_bytes"] is not None:
                        report_id = cache["report_id"]
                        st.download_button(
                            label="Unduh Laporan Inspeksi (PDF)",
                            data=cache["pdf_bytes"],
                            file_name=f"laporan_inspeksi_{report_id}.pdf",
                            mime="application/pdf",
                            type="primary",
                        )
                        st.caption(f"ID Laporan: `{report_id}`")
        else:
            st.info("Silakan unggah citra mobil melalui panel di sebelah kiri untuk memulai analisis.")

if __name__ == "__main__":
    main()
