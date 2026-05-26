import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np
import time
import io
import tempfile
import os

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YOLO Object Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the expected local filename
DEFAULT_MODEL_FILENAME = "best.pt"

# ── Load Model (Handles both Local Path and Bytes) ───────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(source):
    """
    source can be a string path (e.g. 'best.pt') or bytes from an uploaded file
    """
    tmp_path = None
    
    # If it's uploaded bytes, write to a temp file first
    if isinstance(source, bytes):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            tmp.write(source)
            tmp_path = tmp.name
        model_path = tmp_path
    else:
        model_path = source

    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        return model, "ultralytics", tmp_path
    except Exception:
        pass
    try:
        model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path, force_reload=False)
        model.eval()
        return model, "yolov5", tmp_path
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None, str(e), None

# ── Draw Bounding Boxes ────────────────────────────────────────────────────────
PALETTE = [
    "#2ecc71", "#3498db", "#e74c3c", "#9b59b6",
    "#1abc9c", "#f1c40f", "#e67e22", "#34495e"
]

def draw_boxes(image: Image.Image, detections: list) -> Image.Image:
    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    seen_labels = {}
    for label, conf, x1, y1, x2, y2 in detections:
        if label not in seen_labels:
            seen_labels[label] = PALETTE[len(seen_labels) % len(PALETTE)]
        color_hex = seen_labels[label]
        
        r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
        color = (r, g, b, 255)
        fill  = (r, g, b, 35)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3, fill=fill)
        
        tag = f"{label} {conf:.0%}"
        th = 20
        tw = len(tag) * 8 + 8
        draw.rectangle([x1, y1 - th, x1 + tw, y1], fill=color)
        
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
            
        draw.text((x1 + 4, y1 - th + 2), tag, fill=(255, 255, 255, 255), font=font)

    return Image.alpha_composite(img, overlay).convert("RGB")

# ── Inference Engine ──────────────────────────────────────────────────────────
def run_inference(model, backend, image: Image.Image, conf_thresh: float):
    t0 = time.time()
    dets = []

    if backend == "ultralytics":
        results = model.predict(image, conf=conf_thresh, verbose=False)
        r = results[0]
        boxes = r.boxes
        names = r.names
        if boxes is not None and len(boxes):
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls  = int(box.cls[0])
                dets.append((names[cls], conf, int(x1), int(y1), int(x2), int(y2)))
    else:
        results = model(image)
        df = results.pandas().xyxy[0]
        df = df[df["confidence"] >= conf_thresh]
        dets = [
            (row["name"], row["confidence"], int(row["xmin"]), int(row["ymin"]), int(row["xmax"]), int(row["ymax"]))
            for _, row in df.iterrows()
        ]

    elapsed = (time.time() - t0) * 1000
    annotated = draw_boxes(image, dets)
    return annotated, dets, elapsed


# ── UI Layout ──────────────────────────────────────────────────────────────────
st.title("🔍 YOLO Object Detection")
st.markdown("A clean dashboard for local YOLO model deployment and real-time evaluation.")

# ── Sidebar Control Panel
with st.sidebar:
    st.header("⚙️ Configuration")
    
    model = None
    backend = None
    
    # Check if 'best.pt' exists locally in the app directory
    if os.path.exists(DEFAULT_MODEL_FILENAME):
        st.info(f"Using auto-detected model: `{DEFAULT_MODEL_FILENAME}`")
        with st.spinner("Loading weights..."):
            model, backend, _ = load_model(DEFAULT_MODEL_FILENAME)
        if model:
            st.success(f"Active via `{backend}`")
        else:
            st.error(f"Failed to parse `{DEFAULT_MODEL_FILENAME}`: {backend}")
    
    # If no local model is found, display the file uploader fallback
    if model is None:
        uploaded_model = st.file_uploader(
            "Upload Model Weights (.pt)",
            type=["pt"],
            help="Place 'best.pt' in the app directory or upload weights here."
        )
        if uploaded_model is not None:
            with st.spinner("Loading uploaded neural network..."):
                model_bytes = uploaded_model.read()
                model, backend, _ = load_model(model_bytes)
                
            if model is None:
                st.error(f"Failed to load model architecture. Error: {backend}")
                st.stop()
            else:
                st.success(f"Loaded via `{backend}`")
                
    st.divider()
    
    conf_thresh = st.slider(
        "Confidence Threshold", 
        min_value=0.05, 
        max_value=1.00, 
        value=0.25, 
        step=0.05,
        help="Minimum confidence score required to display a detection box."
    )

# ── Main Content Area
if model is None:
    st.info("👈 Please upload your `.pt` model weights file or place `best.pt` in the project folder to begin.")
else:
    uploaded_file = st.file_uploader(
        "Upload Source Image", 
        type=["jpg", "jpeg", "png", "webp", "bmp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        
        with st.spinner("Processing framework inference..."):
            annotated_img, detections, inference_time = run_inference(model, backend, image, conf_thresh)
        
        # ── KPI Metrics Ribbon
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric(label="Total Objects Detected", value=len(detections))
        with m_col2:
            unique_classes = len(set(d[0] for d in detections))
            st.metric(label="Unique Classes Identified", value=unique_classes)
        with m_col3:
            st.metric(label="Inference Velocity", value=f"{inference_time:.1f} ms")
            
        st.divider()

        # ── Image Views Split View
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        with img_col2:
            st.subheader("Model Predictions")
            st.image(annotated_img, use_container_width=True)
            
            buf = io.BytesIO()
            annotated_img.save(buf, format="PNG")
            st.download_button(
                label="💾 Download Rendered Image",
                data=buf.getvalue(),
                file_name=f"detected_{uploaded_file.name}",
                mime="image/png",
                use_container_width=True
            )

        # ── Detections Data Inventory
        if detections:
            st.divider()
            st.subheader("📋 Detailed Breakdown")
            
            table_data = [
                {
                    "Class Index": idx + 1,
                    "Label": det[0],
                    "Confidence Score": f"{det[1]:.2%}",
                    "Box Dimensions (W×H)": f"{det[4] - det[2]} × {det[5] - det[3]} px",
                    "Coordinates [Xmin, Ymin, Xmax, Ymax]": f"[{det[2]}, {det[3]}, {det[4]}, {det[5]}]"
                }
                for idx, det in enumerate(sorted(detections, key=lambda x: -x[1]))
            ]
            st.dataframe(table_data, use_container_width=True, hide_index=True)

    else:
        st.divider()
        st.info("💡 Model is ready! Please drop or upload an image file above to run inference.")
