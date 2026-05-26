from flask import Flask, request, render_template_string
from PIL import Image, ImageDraw, ImageFont
import torch
import os
import io
import time
import base64
import urllib.request

MODEL_URL = "https://https://drive.google.com/drive/u/3/my-drive/best.pt"  # e.g. Google Drive, S3, Hugging Face

if not os.path.exists("best.pt"):
    print("Downloading model...")
    urllib.request.urlretrieve(MODEL_URL, "best.pt")
    print("Done.")

app = Flask(__name__)

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
def load_model():

    try:
        from ultralytics import YOLO

        model = YOLO(MODEL_PATH)

        return model, "ultralytics"

    except Exception as e:

        print("MODEL LOAD ERROR:", e)

        return None, None

# ---------------------------------------------------
# COLORS
# ---------------------------------------------------
COLORS = [
    "#000000",
    "#444444",
    "#666666",
    "#888888"
]

# ---------------------------------------------------
# DRAW BOXES
# ---------------------------------------------------

def draw_boxes(image, detections):

    img = image.copy().convert("RGB")

    draw = ImageDraw.Draw(img)

    GREEN = "#00aa00"

    for det in detections:

        label, conf, x1, y1, x2, y2 = det

        # THICKER GREEN BOX
        draw.rectangle(
            [x1, y1, x2, y2],
            outline=GREEN,
            width=6
        )

        # LARGER LABEL
        text = f"{label} {conf:.2f}"

        try:
            font = ImageFont.truetype(
                "arial.ttf",
                48
            )

        except:
            font = ImageFont.load_default()

        # TEXT SIZE
        bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # LABEL BACKGROUND
        draw.rectangle(
            [
                x1,
                y1 - text_height - 16,
                x1 + text_width + 20,
                y1
            ],
            fill=GREEN
        )

        # TEXT
        draw.text(
            (
                x1 + 10,
                y1 - text_height - 8
            ),
            text,
            fill="white",
            font=font
        )

    return img

# ---------------------------------------------------
# RUN INFERENCE
# ---------------------------------------------------
def run_inference(model, image, conf_thresh):

    start = time.time()

    detections = []

    results = model.predict(
        image,
        conf=conf_thresh,
        verbose=False
    )

    r = results[0]

    if r.boxes is not None:

        for box in r.boxes:

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            label = r.names[cls]

            detections.append(
                (
                    label,
                    conf,
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2)
                )
            )

    elapsed = (
        time.time() - start
    ) * 1000

    annotated = draw_boxes(
        image,
        detections
    )

    return annotated, detections, elapsed

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------
model, backend = load_model()

# ---------------------------------------------------
# LOGO
# ---------------------------------------------------
LOGO_PATH = r"/mnt/data/ChatGPT Image May 25, 2026, 03_30_36 PM.png"

logo_b64 = ""

if os.path.exists(LOGO_PATH):

    with open(LOGO_PATH, "rb") as f:

        logo_b64 = base64.b64encode(
            f.read()
        ).decode()

# ---------------------------------------------------
# HTML TEMPLATE
# ---------------------------------------------------
HTML = """

<!DOCTYPE html>

<html>

<head>

<title>YOLO Detector</title>

<style>

*{
    margin:0;
    padding:0;
    box-sizing:border-box;
    font-family:Arial,sans-serif;
}

body{
    background:white;
    color:black;
    padding:40px;
}

.container{
    max-width:1200px;
    margin:auto;
}

.logo-box{
    text-align:center;
    margin-bottom:25px;
}

.logo-box img{
    width:220px;
    max-width:100%;
}

h1{
    font-size:40px;
    font-weight:500;
    margin-bottom:10px;
    text-align:center;
}

.subtitle{
    color:#666;
    margin-bottom:40px;
    text-align:center;
}

.card{
    border:1px solid #ddd;
    padding:25px;
    margin-bottom:30px;
}

form{
    display:flex;
    flex-direction:column;
    gap:20px;
}

input[type=file]{
    border:1px solid #ccc;
    padding:12px;
}

input[type=range]{
    width:100%;
}

.slider-value{
    color:#666;
    font-size:14px;
}

button{
    background:black;
    color:white;
    border:none;
    padding:14px;
    cursor:pointer;
    font-size:15px;
}

button:hover{
    opacity:0.9;
}

.metrics{
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:20px;
    margin-bottom:30px;
}

.metric{
    border:1px solid #ddd;
    padding:25px;
}

.metric h2{
    font-size:42px;
    font-weight:400;
    margin-bottom:8px;
}

.metric p{
    color:#666;
}

.images{
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:25px;
}

.image-box{
    border:1px solid #ddd;
    padding:20px;
}

.image-box h3{
    margin-bottom:15px;
    font-weight:500;
}

.image-box img{
    width:100%;
}

.table-box{
    border:1px solid #ddd;
    padding:25px;
    margin-top:30px;
}

table{
    width:100%;
    border-collapse:collapse;
    margin-top:20px;
}

th{
    background:#f5f5f5;
    text-align:left;
    padding:14px;
}

td{
    padding:14px;
    border-bottom:1px solid #eee;
}

.error{
    color:red;
    margin-top:20px;
}

@media(max-width:900px){

    .images{
        grid-template-columns:1fr;
    }

    .metrics{
        grid-template-columns:1fr;
    }

}

</style>

</head>

<body>

<div class="container">

    <div class="logo-box">

        <img
            src="data:image/png;base64,{{ logo }}"
        >

    </div>

    <h1>
    YOLO Object Detector
    </h1>

    <div class="subtitle">
    Local object detection using YOLO
    </div>

    <div class="card">

        <form method="POST" enctype="multipart/form-data">

            <div>

                <label>
                Upload Image
                </label>

            </div>

            <input
                type="file"
                name="image"
                required
            >

            <div>

                <label>
                Confidence Threshold
                </label>

            </div>

            <input
                type="range"
                name="conf"
                id="confSlider"
                min="0.05"
                max="1"
                step="0.05"
                value="0.25"
                oninput="updateSlider(this.value)"
            >

            <div class="slider-value">

                Current Value:
                <span id="sliderText">
                0.25
                </span>

            </div>

            <button type="submit">

                Run Detection

            </button>

        </form>

        {% if model_loaded == False %}

        <div class="error">

            best.pt not found in project folder

        </div>

        {% endif %}

    </div>

    {% if result %}

    <div class="metrics">

        <div class="metric">

            <h2>
            {{ total }}
            </h2>

            <p>
            Total Objects
            </p>

        </div>

        <div class="metric">

            <h2>
            {{ unique }}
            </h2>

            <p>
            Unique Classes
            </p>

        </div>

        <div class="metric">

            <h2>
            {{ speed }}
            </h2>

            <p>
            Inference Time
            </p>

        </div>

    </div>

    <div class="images">

        <div class="image-box">

            <h3>
            Original Image
            </h3>

            <img
                src="data:image/png;base64,{{ original }}"
            >

        </div>

        <div class="image-box">

            <h3>
            Detection Result
            </h3>

            <img
                src="data:image/png;base64,{{ detected }}"
            >

        </div>

    </div>

    <div class="table-box">

        <h2>
        Detection Results
        </h2>

        <table>

            <tr>

                <th>
                Class
                </th>

                <th>
                Confidence
                </th>

                <th>
                Coordinates
                </th>

            </tr>

            {% for d in detections %}

            <tr>

                <td>
                {{ d[0] }}
                </td>

                <td>
                {{ '%.2f'|format(d[1] * 100) }}%
                </td>

                <td>
                [{{ d[2] }}, {{ d[3] }}, {{ d[4] }}, {{ d[5] }}]
                </td>

            </tr>

            {% endfor %}

        </table>

    </div>

    {% endif %}

</div>

<script>

function updateSlider(value){

    document.getElementById(
        "sliderText"
    ).innerText = value;

}

</script>

</body>

</html>

"""

# ---------------------------------------------------
# ROUTE
# ---------------------------------------------------
@app.route("/", methods=["GET", "POST"])

def index():

    if request.method == "POST":

        if model is None:

            return render_template_string(
                HTML,
                model_loaded=False,
                result=False,
                logo=logo_b64
            )

        file = request.files["image"]

        conf = float(
            request.form.get(
                "conf",
                0.25
            )
        )

        image = Image.open(
            file.stream
        ).convert("RGB")

        annotated, detections, speed = run_inference(
            model,
            image,
            conf
        )

        original_buffer = io.BytesIO()

        image.save(
            original_buffer,
            format="PNG"
        )

        detected_buffer = io.BytesIO()

        annotated.save(
            detected_buffer,
            format="PNG"
        )

        original_b64 = base64.b64encode(
            original_buffer.getvalue()
        ).decode()

        detected_b64 = base64.b64encode(
            detected_buffer.getvalue()
        ).decode()

        return render_template_string(

            HTML,

            model_loaded=True,

            result=True,

            original=original_b64,

            detected=detected_b64,

            detections=detections,

            total=len(detections),

            unique=len(
                set(d[0] for d in detections)
            ),

            speed=f"{speed:.1f} ms",

            logo=logo_b64

        )

    return render_template_string(

        HTML,

        model_loaded=(
            model is not None
        ),

        result=False,

        logo=logo_b64

    )

# ---------------------------------------------------
# RUN APP
# ---------------------------------------------------
if __name__ == "__main__":

    app.run(
        debug=True
    )
