
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DigitAI | Handwriting Recognition",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

MODEL_PATH = Path(__file__).parent / "handwritten_digit_model.keras"


# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, #252050 0%, transparent 38%),
        radial-gradient(ellipse at 95% 30%, #102b4c 0%, transparent 35%),
        #080c18;
    color: #f4f6ff;
    font-family: 'Inter', sans-serif;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

#MainMenu, footer {
    visibility: hidden;
}

.brand {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #ffffff;
}

.brand span {
    color: #a78bfa;
}

.top-tag {
    border: 1px solid #34405e;
    background: #171c30;
    border-radius: 30px;
    padding: 7px 13px;
    color: #b8c5e4;
    font-size: 12px;
    text-align: center;
}

.hero {
    text-align: center;
    padding: 50px 10px 26px;
}

.eyebrow {
    color: #c4b5fd;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    color: #ffffff;
    font-size: clamp(2.1rem, 5vw, 3.7rem);
    font-weight: 800;
    line-height: 1.13;
    letter-spacing: -2px;
    margin: 18px 0;
}

.gradient-text {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero p {
    color: #a7b2cb;
    font-size: 16px;
    line-height: 1.8;
    max-width: 650px;
    margin: auto;
}

.section-label {
    color: #e6eaff;
    font-size: 19px;
    font-weight: 700;
    margin: 12px 0;
}

.muted {
    color: #9ba9c5;
    font-size: 13px;
    line-height: 1.7;
}

.panel {
    background: rgba(19, 26, 46, 0.86);
    border: 1px solid #2a3553;
    border-radius: 18px;
    padding: 24px;
    min-height: 150px;
}

.result-panel {
    background: linear-gradient(145deg, #201b42, #132b43);
    border: 1px solid #6454a4;
    border-radius: 20px;
    padding: 28px 20px;
    text-align: center;
}

.result-digit {
    font-size: 100px;
    font-weight: 800;
    line-height: 1.2;
    background: linear-gradient(135deg, #c4b5fd, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.result-caption {
    color: #d6dafa;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.metric-box {
    background: #11182a;
    border: 1px solid #2c3855;
    border-radius: 14px;
    padding: 16px;
    margin-top: 14px;
}

div[data-testid="stFileUploader"] {
    background: #10172a;
    border: 1px dashed #6b63b8;
    border-radius: 16px;
    padding: 14px;
}

div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] p {
    color: #e2e8f8 !important;
}

.stButton > button {
    background: linear-gradient(100deg, #7c3aed, #4263eb);
    color: white;
    border: none;
    border-radius: 12px;
    font-weight: 700;
    min-height: 48px;
    width: 100%;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border: 1px solid #b7a6ff;
    color: white;
    filter: brightness(1.12);
}

div[data-testid="stImage"] img {
    border-radius: 12px;
}

div[data-testid="stProgress"] > div {
    background: #29344d;
}

hr {
    border-color: #28324b;
    margin: 28px 0;
}

.footer {
    text-align: center;
    color: #7785a4;
    font-size: 12px;
    padding-top: 30px;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- MODEL LOADING ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ---------------- IMAGE PREPROCESSING ----------------
def preprocess_image(uploaded_image):
    image = uploaded_image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(2.0)

    # MNIST uses a dark background with a light digit.
    # Invert typical photos of dark ink on white paper.
    if np.asarray(image).mean() > 127:
        image = ImageOps.invert(image)

    image = image.point(lambda pixel: 255 if pixel > 60 else 0)

    bbox = image.getbbox()
    if bbox:
        image = image.crop(bbox)

    image.thumbnail((20, 20))

    canvas = Image.new("L", (28, 28), 0)
    x_offset = (28 - image.width) // 2
    y_offset = (28 - image.height) // 2
    canvas.paste(image, (x_offset, y_offset))

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    array = array.reshape(1, 28, 28, 1)

    return array, canvas


# ---------------- HEADER ----------------
header_left, header_right = st.columns([3, 1])

with header_left:
    st.markdown(
        '<div class="brand">✦ Digit<span>AI</span></div>',
        unsafe_allow_html=True
    )

with header_right:
    st.markdown(
        '<div class="top-tag">● CNN MODEL &nbsp;|&nbsp; READY</div>',
        unsafe_allow_html=True
    )


# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <div class="eyebrow">✦ &nbsp; POWERED BY DEEP LEARNING</div>
    <h1>Every stroke tells a story.<br>
    <span class="gradient-text">Let AI read it.</span></h1>
    <p>
        Turn handwritten digits into intelligent predictions.
        Upload your image and watch a Convolutional Neural Network
        recognize the number in seconds.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------------- QUICK INFO ----------------
m1, m2, m3 = st.columns(3)

with m1:
    st.markdown("""
    <div class="panel">
        <div style="font-size:25px">✍️</div>
        <b>Easy Upload</b>
        <p class="muted">Upload a clear handwritten digit from 0 to 9.</p>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown("""
    <div class="panel">
        <div style="font-size:25px">🧠</div>
        <b>CNN Intelligence</b>
        <p class="muted">Deep learning analyzes the shape and patterns.</p>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown("""
    <div class="panel">
        <div style="font-size:25px">📊</div>
        <b>Visual Results</b>
        <p class="muted">Explore the predicted digit and class probabilities.</p>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-label">01 &nbsp; Upload your image</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="muted">For best results, use one large, dark digit '
        'on a plain, light background.</p>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Choose a PNG or JPG image",
        type=["png", "jpg", "jpeg"],
        help="Upload an image containing a single handwritten digit."
    )

    if uploaded_file:
        original_image = Image.open(uploaded_file)

        st.image(
            original_image,
            caption="Your uploaded image",
            use_container_width=True
        )

        predict_clicked = st.button(
            "✦  Recognize My Digit",
            type="primary",
            use_container_width=True
        )
    else:
        predict_clicked = False
        st.markdown("""
        <div class="panel" style="text-align:center;margin-top:12px">
            <div style="font-size:42px">↥</div>
            <b>Your image goes here</b>
            <p class="muted">PNG or JPG · One digit per image</p>
        </div>
        """, unsafe_allow_html=True)


# Keep prediction visible after the button is clicked
if "digit_result" not in st.session_state:
    st.session_state.digit_result = None

if predict_clicked and uploaded_file:
    try:
        model = load_model()
        processed_array, processed_image = preprocess_image(
            original_image
        )

        probabilities = model.predict(processed_array, verbose=0)[0]

        st.session_state.digit_result = {
            "digit": int(np.argmax(probabilities)),
            "confidence": float(np.max(probabilities)) * 100,
            "probabilities": probabilities.tolist(),
            "processed_image": processed_image
        }

    except Exception as error:
        st.session_state.digit_result = None
        st.error(f"Prediction failed: {error}")


# ---------------- RESULTS ----------------
with right:
    st.markdown(
        '<div class="section-label">02 &nbsp; AI prediction</div>',
        unsafe_allow_html=True
    )

    result = st.session_state.digit_result

    if result is not None:
        st.markdown(f"""
        <div class="result-panel">
            <div class="result-caption">PREDICTED DIGIT</div>
            <div class="result-digit">{result["digit"]}</div>
            <div style="color:#c4b5fd;font-weight:600">
                ✦ Recognition complete
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="metric-box"><b>Model confidence</b></div>',
            unsafe_allow_html=True
        )

        st.progress(
            min(max(result["confidence"] / 100, 0.0), 1.0)
        )

        st.markdown(
            f'<p class="muted">{result["confidence"]:.2f}% '
            'probability assigned to the predicted class.</p>',
            unsafe_allow_html=True
        )

        with st.expander("View processed 28 × 28 image"):
            st.image(
                result["processed_image"],
                caption="Image given to the CNN model",
                width=180
            )

    else:
        st.markdown("""
        <div class="result-panel" style="padding:45px 15px">
            <div style="font-size:52px">✧</div>
            <div class="result-caption">AWAITING YOUR IMAGE</div>
            <p class="muted" style="margin-top:15px">
                Your prediction will appear here
                after you click Recognize My Digit.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ---------------- PROBABILITY CHART ----------------
result = st.session_state.digit_result

if result is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(
        '<div class="section-label">03 &nbsp; Prediction analytics</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="muted">Compare the model probabilities for all ten digits. '
        'Higher values indicate stronger model preference, not guaranteed '
        'correctness.</p>',
        unsafe_allow_html=True
    )

    chart_data = pd.DataFrame({
        "Digit": [str(i) for i in range(10)],
        "Probability (%)": [
            round(p * 100, 2) for p in result["probabilities"]
        ]
    }).set_index("Digit")

    st.bar_chart(
        chart_data,
        y="Probability (%)",
        height=300
    )


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <b>DigitAI</b> &nbsp;·&nbsp; Built with Python, TensorFlow,
    CNN & Streamlit<br><br>
    An educational machine learning project. Predictions can be incorrect,
    especially for handwriting unlike the training data.
</div>
""", unsafe_allow_html=True)
