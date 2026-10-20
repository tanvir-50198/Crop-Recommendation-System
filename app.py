"""
Crop Studio — Professional Crop Recommendation Dashboard (single-file Streamlit app)
------------------------------------------------------------------------------------
Run:
    streamlit run app.py

Required files in the SAME folder:
    best_crop_model.pkl
    crop_label_encoder.pkl

Install:
    pip install streamlit pandas numpy scikit-learn xgboost

Pages: Recommend | Batch | History | About | Feedback
"""

import inspect
import io
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import joblib

st.set_page_config(
    page_title="Crop Studio · AI Crop Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def _stretch(fn):
    return {"width": "stretch"} if "width" in inspect.signature(fn).parameters else {"use_container_width": True}


S_BTN, S_DL, S_SUB, S_DF = (_stretch(st.button), _stretch(st.download_button),
                            _stretch(st.form_submit_button), _stretch(st.dataframe))

APP_DIR = Path(__file__).resolve().parent
MODEL_FILE = APP_DIR / "best_crop_model.pkl"
ENCODER_FILE = APP_DIR / "crop_label_encoder.pkl"
FEEDBACK_FILE = APP_DIR / "feedback.json"
PAGES = ["Recommend", "Batch", "History", "About", "Feedback"]
MAX_MB = 10
DATASET_ROWS = 57000  # from the training notebook

# label, unit, min, max, default, step
NUMERIC_SPEC = {
    "SOIL_PH":           ("Soil pH", "", 5.0, 9.0, 6.6, 0.1),
    "TEMP":              ("Temperature", "°C", 5.0, 47.0, 25.0, 0.5),
    "RELATIVE_HUMIDITY": ("Relative humidity", "%", 15.0, 100.0, 78.0, 1.0),
    "WATERREQUIRED":     ("Water available", "mm", 330.0, 2500.0, 930.0, 10.0),
    "CROPDURATION":      ("Crop duration", "days", 20.0, 330.0, 100.0, 1.0),
    "N":                 ("Nitrogen (N)", "kg/ha", 20.0, 200.0, 80.0, 1.0),
    "P":                 ("Phosphorus (P)", "kg/ha", 20.0, 100.0, 48.0, 1.0),
    "K":                 ("Potassium (K)", "kg/ha", 20.0, 150.0, 50.0, 1.0),
}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_FULL = dict(zip(MONTHS, ["January", "February", "March", "April", "May", "June", "July",
                               "August", "September", "October", "November", "December"]))
SEASON_LABEL = {"kharif": "Kharif (monsoon)", "rabi": "Rabi (winter)", "zaid": "Zaid (summer)"}
WATER_LABEL = {"irrigated": "Irrigated", "rainfed": "Rain-fed"}

PRESETS = {
    "🌾 Monsoon paddy": dict(SOIL="Alluvial soil", WATER_SOURCE="irrigated", SEASON="kharif", SOWN="Jun",
                             HARVESTED="Sep", SOIL_PH=7.6, TEMP=27.0, RELATIVE_HUMIDITY=74.0,
                             WATERREQUIRED=2460.0, CROPDURATION=117.0, N=82.0, P=41.0, K=42.0),
    "❄️ Winter cereal": dict(SOIL="Loamy soil", WATER_SOURCE="irrigated", SEASON="rabi", SOWN="Nov",
                            HARVESTED="Mar", SOIL_PH=6.8, TEMP=18.0, RELATIVE_HUMIDITY=60.0,
                            WATERREQUIRED=450.0, CROPDURATION=120.0, N=120.0, P=60.0, K=40.0),
    "☀️ Dry summer": dict(SOIL="Sandy soil", WATER_SOURCE="irrigated", SEASON="zaid", SOWN="Mar",
                         HARVESTED="Jun", SOIL_PH=6.5, TEMP=34.0, RELATIVE_HUMIDITY=55.0,
                         WATERREQUIRED=600.0, CROPDURATION=90.0, N=60.0, P=40.0, K=60.0),
}

CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "wheat": "🌾", "sugarcane": "🎋", "groundnut": "🥜", "onion": "🧅",
    "small onion": "🧅", "carrot": "🥕", "cabbage": "🥬", "cauliflower": "🥦", "brinjal": "🍆",
    "capsicum": "🫑", "chillies": "🌶️", "cucumber": "🥒", "tomato": "🍅", "potato": "🥔",
    "pumpkin": "🎃", "peas": "🫛", "french bean": "🫘", "soyabean": "🫘", "blackgram": "🫘",
    "greengram": "🫘", "redgram": "🫘", "bengalgram": "🫘", "cowpea": "🫘", "horsegram": "🫘",
    "muskmelon": "🍈", "watermelon": "🍉", "banana": "🍌", "sunflower": "🌻", "coconut": "🥥",
    "mango": "🥭", "sorghum": "🌾", "ragi": "🌾", "pearl millet": "🌾", "samai": "🌾",
    "cotton": "☁️", "jute": "🌿", "bhendi": "🌿", "radish": "🌱", "beetroot": "🌱",
}


def emoji_for(name):
    return CROP_EMOJI.get(str(name).strip().lower(), "🌱")


def H(html):
    """Collapse multi-line HTML so Markdown never treats indented lines as a code block."""
    return "".join(line.strip() for line in html.splitlines())


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600&display=swap');

:root{
  --forest:#173B2A; --leaf:#2F7044; --leaf-2:#3E8A57; --mint:#E3F2E6; --lime:#D6EB7A;
  --bg:#F4F6EF; --paper:#FFFFFF; --ink:#14241B; --muted:#5E6E63; --line:#E0E6D9;
  --amber:#E2A93B; --shadow:0 1px 2px rgba(23,59,42,.05), 0 8px 24px rgba(23,59,42,.06);
}

html, body, .stApp, [data-testid="stAppViewContainer"]{
  background:var(--bg) !important; color:var(--ink);
  font-family:'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif; color-scheme:light;
}
[data-testid="stHeader"]{ background:transparent !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"]{ display:none !important; }
.block-container{ padding-top:.8rem !important; padding-bottom:3rem; max-width:1240px; }
h1,h2,h3,h4{ font-family:'Plus Jakarta Sans','Inter',sans-serif; }

/* ---------- navbar ---------- */
.st-key-navbar{
  background:linear-gradient(135deg,var(--forest),var(--leaf)); border-radius:18px;
  padding:.55rem .9rem; margin:0 0 1.2rem 0; box-shadow:var(--shadow);
}
.st-key-navbar [data-testid="stHorizontalBlock"]{ flex-wrap:wrap; gap:.3rem; }
.st-key-navbar [data-testid="stColumn"]{ min-width:0; }
.st-key-navbar button{
  min-height:40px; padding:.3rem .6rem !important; border-radius:999px !important;
  font-size:.92rem !important; font-weight:600 !important; white-space:nowrap; box-shadow:none !important;
}
.st-key-navbar button[data-testid="stBaseButton-secondary"]{ background:transparent !important; border:1px solid transparent !important; }
.st-key-navbar button[data-testid="stBaseButton-secondary"] *{ color:#CFE5D3 !important; }
.st-key-navbar button[data-testid="stBaseButton-secondary"]:hover{ background:rgba(255,255,255,.1) !important; }
.st-key-navbar button[data-testid="stBaseButton-primary"]{ background:rgba(255,255,255,.2) !important; border:1px solid rgba(255,255,255,.25) !important; }
.st-key-navbar button[data-testid="stBaseButton-primary"] *{ color:#fff !important; }
.brand{ display:flex; align-items:center; gap:.65rem; }
.brand-mark{ width:38px; height:38px; border-radius:11px; background:rgba(255,255,255,.14); display:flex; align-items:center; justify-content:center; }
.brand-name{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.15rem; color:#fff; line-height:1.1; }
.brand-sub{ font-size:.74rem; color:#BBD6C1; }

/* ---------- hero ---------- */
.hero{
  position:relative; overflow:hidden; border-radius:22px; padding:2rem 2.2rem; margin-bottom:1.3rem;
  background:radial-gradient(900px 300px at 85% -10%, #DDF1D8 0%, transparent 60%),
             linear-gradient(135deg,#EAF6E8 0%,#F7FAEF 100%);
  border:1px solid var(--line);
  display:grid; grid-template-columns:1.25fr 1fr; gap:2rem; align-items:center;
}
.hero::before{ content:"🌿"; position:absolute; right:-10px; top:-18px; font-size:7rem; opacity:.07; transform:rotate(18deg); }
.pill{ display:inline-flex; align-items:center; gap:.4rem; background:#fff; border:1px solid var(--line);
       border-radius:999px; padding:.3rem .8rem; font-size:.8rem; font-weight:600; color:var(--leaf); box-shadow:var(--shadow); }
.dot{ width:7px; height:7px; border-radius:50%; background:var(--leaf-2); display:inline-block; }
.hero h1{ font-weight:800; font-size:clamp(1.9rem,3.6vw,2.9rem); line-height:1.08; letter-spacing:-.025em; margin:.9rem 0 .6rem 0; color:var(--forest); }
.hero h1 em{ font-style:normal; background:linear-gradient(90deg,var(--leaf),#1B9A6C); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero p{ color:var(--muted); font-size:1.02rem; line-height:1.6; max-width:52ch; margin:0; }
.stats{ display:grid; grid-template-columns:repeat(2,1fr); gap:.8rem; }
.stat{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:1rem 1.1rem; box-shadow:var(--shadow); }
.stat .ic{ width:34px; height:34px; border-radius:10px; background:var(--mint); display:flex; align-items:center; justify-content:center; font-size:1.05rem; margin-bottom:.55rem; }
.stat b{ display:block; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.55rem; font-weight:800; color:var(--forest); line-height:1.1; }
.stat span{ font-size:.82rem; color:var(--muted); }

/* ---------- section cards ---------- */
div[data-testid="stForm"]{ border:none !important; padding:0 !important; background:transparent !important; }
[class*="st-key-sec_"]{
  background:var(--paper) !important; border:1px solid var(--line) !important; border-radius:18px !important;
  padding:.6rem .8rem .5rem .8rem; margin-bottom:.8rem; box-shadow:var(--shadow);
}
.sec-head{ display:flex; align-items:center; gap:.75rem; margin-bottom:.5rem; }
.sec-ic{ width:38px; height:38px; border-radius:11px; background:var(--mint); display:flex; align-items:center; justify-content:center; font-size:1.15rem; flex:0 0 auto; }
.sec-head h3{ font-weight:700; font-size:1.05rem; margin:0; color:var(--ink); line-height:1.2; }
.sec-head p{ color:var(--muted); font-size:.85rem; margin:0; }
.lbl{ font-weight:700; font-size:.9rem; color:var(--ink); margin:.1rem 0 .5rem 0; }

/* ---------- widgets ---------- */
label, [data-testid="stWidgetLabel"] p{ color:var(--ink) !important; font-weight:600 !important; font-size:.88rem !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"], [data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"]{ color:var(--muted) !important; }
[data-testid="stSlider"] [data-testid="stThumbValue"]{ font-weight:700; }
[data-testid="stSlider"]{ filter:hue-rotate(118deg) saturate(.62) brightness(.78); }
[data-baseweb="select"] > div{ background:#fff !important; border-color:var(--line) !important; border-radius:12px !important; min-height:44px; }
[data-baseweb="select"] *{ color:var(--ink) !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li{ background:#fff !important; color:var(--ink) !important; }
[data-baseweb="popover"] li:hover{ background:var(--mint) !important; }
[data-baseweb="input"], [data-baseweb="textarea"]{ background:#fff !important; border-radius:12px !important; }
.st-key-feedback_wrap input, .st-key-feedback_wrap textarea{ background:#fff !important; color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; }

.stFormSubmitButton, .stFormSubmitButton > div{ width:100% !important; }
.stFormSubmitButton button[data-testid="stBaseButton-primaryFormSubmit"], .stButton button[data-testid="stBaseButton-primary"]{
  width:100% !important; border:0; border-radius:14px; padding:.95rem 1rem; min-height:54px;
  background:linear-gradient(135deg,var(--leaf),#1F5A36) !important; color:#fff !important;
  font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:1.05rem;
  box-shadow:0 8px 20px rgba(47,112,68,.28); transition:transform .15s ease, box-shadow .15s ease;
}
.stFormSubmitButton button[data-testid="stBaseButton-primaryFormSubmit"] *{ color:#fff !important; }
.stFormSubmitButton button[data-testid="stBaseButton-primaryFormSubmit"]:hover{ transform:translateY(-1px); box-shadow:0 12px 26px rgba(47,112,68,.34); }
.stFormSubmitButton button:focus-visible, .stButton button:focus-visible{ outline:3px solid var(--lime); outline-offset:2px; }

.stButton button[data-testid="stBaseButton-secondary"], .stDownloadButton button, .stFormSubmitButton button[data-testid="stBaseButton-secondaryFormSubmit"]{
  background:#fff !important; color:var(--ink) !important; border:1px solid var(--line) !important; border-radius:12px !important;
  padding:.5rem .9rem; font-weight:600; font-size:.9rem; min-height:42px;
}
.stButton button[data-testid="stBaseButton-secondary"] *, .stDownloadButton button *{ color:var(--ink) !important; }
.stButton button[data-testid="stBaseButton-secondary"]:hover, .stDownloadButton button:hover{ border-color:var(--leaf) !important; background:#F2F8EF !important; }
.st-key-navbar button[data-testid="stBaseButton-secondary"]{ background:transparent !important; border:1px solid transparent !important; }
.st-key-navbar button[data-testid="stBaseButton-secondary"] *{ color:#CFE5D3 !important; }

/* ---------- result ---------- */
.res{ background:var(--paper); border:1px solid var(--line); border-radius:22px; overflow:hidden; box-shadow:var(--shadow); }
.res-top{ background:linear-gradient(135deg,var(--forest),var(--leaf)); padding:1.5rem 1.6rem; color:#fff;
          display:flex; justify-content:space-between; align-items:center; gap:1rem; position:relative; }
.res-top .tag{ font-size:.72rem; letter-spacing:.12em; font-weight:700; color:var(--lime); text-transform:uppercase; }
.res-top .crop{ display:flex; align-items:center; gap:.7rem; margin:.35rem 0 .5rem 0;
                font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:clamp(1.9rem,3.2vw,2.6rem);
                line-height:1.05; letter-spacing:-.02em; text-transform:capitalize; }
.res-top .crop .em{ font-size:2.3rem; background:rgba(255,255,255,.14); border-radius:16px; padding:.15rem .5rem; }
.res-top .sub{ color:#CFE5D3; font-size:.88rem; }
.gauge{ position:relative; width:112px; height:112px; flex:0 0 auto; }
.gauge svg{ transform:rotate(-90deg); }
.gauge .g-val{ position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center; }
.gauge .g-val b{ font-family:'Plus Jakarta Sans',sans-serif; font-size:1.55rem; font-weight:800; line-height:1; }
.gauge .g-val span{ font-size:.66rem; color:#CFE5D3; letter-spacing:.08em; text-transform:uppercase; margin-top:.15rem; }
.res-body{ padding:1.2rem 1.6rem 1.5rem 1.6rem; }
.res-body h4{ font-size:.95rem; font-weight:700; margin:0 0 .8rem 0; color:var(--ink); }
.rank{ display:grid; grid-template-columns:2rem 1.6rem 7rem 1fr 3.4rem; align-items:center; gap:.55rem; padding:.5rem .6rem; border-radius:12px; margin-bottom:.25rem; }
.rank.first{ background:var(--mint); }
.rank .n{ width:24px; height:24px; border-radius:50%; background:#EDF1E8; color:var(--muted); font-size:.75rem; font-weight:700; display:flex; align-items:center; justify-content:center; }
.rank.first .n{ background:var(--leaf); color:#fff; }
.rank .e{ font-size:1.15rem; }
.rank .name{ font-weight:600; font-size:.92rem; text-transform:capitalize; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.track{ height:9px; background:#E7EDE2; border-radius:99px; overflow:hidden; }
.fill{ height:100%; border-radius:99px; background:linear-gradient(90deg,#8FC48B,var(--leaf-2)); }
.rank.first .fill{ background:linear-gradient(90deg,var(--leaf-2),var(--forest)); }
.rank .pct{ text-align:right; font-variant-numeric:tabular-nums; font-size:.88rem; color:var(--muted); font-weight:600; }
.why{ margin-top:1rem; padding:.9rem 1rem; border-radius:14px; background:#F7FAF2; border:1px solid var(--line); font-size:.9rem; line-height:1.55; color:#34463B; }
.why b{ color:var(--forest); }
.warn{ margin-top:.8rem; padding:.8rem 1rem; border-radius:14px; background:#FBF4DE; border:1px solid #EBDDA6; color:#5B4A12; font-size:.9rem; line-height:1.5; }
.chips{ display:flex; flex-wrap:wrap; gap:.4rem; margin-top:1rem; }
.chip{ background:#F1F5EC; border:1px solid var(--line); border-radius:999px; padding:.22rem .7rem; font-size:.8rem; }
.chip b{ color:var(--muted); font-weight:600; margin-right:.3rem; }

.empty{ background:var(--paper); border:1.5px dashed #BCC9B3; border-radius:22px; padding:2.4rem 1.8rem; text-align:center; color:var(--muted); }
.empty .big{ font-size:2.6rem; margin-bottom:.4rem; }
.empty h3{ font-weight:700; color:var(--ink); font-size:1.2rem; margin:0 0 .5rem 0; }
.empty p{ margin:0 auto; line-height:1.55; max-width:38ch; font-size:.93rem; }

/* ---------- snapshot tiles ---------- */
.snap-title{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:700; font-size:1.05rem; margin:1.4rem 0 .7rem 0; }
.snap{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.7rem; }
.tile{ background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:.9rem 1rem; box-shadow:var(--shadow); }
.tile .k{ font-size:.78rem; color:var(--muted); font-weight:600; }
.tile .v{ font-family:'Plus Jakarta Sans',sans-serif; font-size:1.35rem; font-weight:800; color:var(--forest); margin:.15rem 0 .5rem 0; }
.tile .v small{ font-size:.75rem; font-weight:600; color:var(--muted); margin-left:.2rem; }
.tile .mini{ height:6px; background:#E7EDE2; border-radius:99px; overflow:hidden; }
.tile .mini i{ display:block; height:100%; background:var(--leaf-2); border-radius:99px; }

/* ---------- misc ---------- */
.page-title{ font-family:'Plus Jakarta Sans',sans-serif; font-weight:800; font-size:1.8rem; letter-spacing:-.02em; margin:.2rem 0 .25rem 0; color:var(--forest); }
.page-sub{ color:var(--muted); margin:0 0 1.2rem 0; max-width:64ch; }
.kpi{ background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:1rem 1.2rem; box-shadow:var(--shadow); }
.kpi b{ display:block; font-family:'Plus Jakarta Sans',sans-serif; font-size:1.6rem; line-height:1.1; color:var(--forest); text-transform:capitalize; }
.kpi span{ color:var(--muted); font-size:.84rem; }
.card{ background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1.2rem; margin-bottom:.9rem; min-height:128px; box-shadow:var(--shadow); }
.card .ic{ width:36px; height:36px; border-radius:10px; background:var(--mint); display:flex; align-items:center; justify-content:center; margin-bottom:.6rem; font-size:1.05rem; }
.card h4{ margin:0 0 .3rem 0; font-size:1rem; font-weight:700; }
.card p{ margin:0; color:var(--muted); line-height:1.5; font-size:.9rem; }
.stepper{ display:flex; align-items:flex-start; gap:.4rem; background:var(--paper); border:1px solid var(--line); border-radius:16px; padding:1.1rem 1rem; margin:.4rem 0 1.1rem 0; flex-wrap:wrap; box-shadow:var(--shadow); }
.step{ flex:1; min-width:150px; text-align:center; }
.circle{ width:40px; height:40px; border-radius:50%; margin:0 auto .5rem auto; display:flex; align-items:center; justify-content:center; font-weight:700; background:#EDF1E8; color:var(--leaf); border:2px solid #CBD7C4; }
.s-active .circle{ background:var(--leaf); color:#fff; border-color:var(--leaf); box-shadow:0 0 0 5px rgba(47,112,68,.16); }
.s-done .circle{ background:var(--lime); color:var(--forest); border-color:var(--lime); }
.step-title{ font-weight:700; font-size:.95rem; } .step-desc{ color:var(--muted); font-size:.84rem; }
.line{ flex:0 0 36px; height:2px; background:#CBD7C4; margin-top:20px; } .line.done{ background:#9DBB4F; }
.crop-chip{ display:inline-block; margin:.15rem; padding:.22rem .7rem; border-radius:999px; background:#fff; border:1px solid var(--line); font-size:.85rem; text-transform:capitalize; }
code{ color:var(--leaf) !important; background:var(--mint) !important; padding:.1rem .35rem; border-radius:6px; }
[data-testid="stDataFrame"]{ border:1px solid var(--line); border-radius:14px; overflow:hidden; }
[data-testid="stFileUploader"] section{ background:#fff !important; border:1.5px dashed #9DB394 !important; border-radius:16px !important; }
[data-testid="stFileUploader"] section *{ color:var(--ink) !important; }
[data-testid="stFileUploader"] section button{ background:var(--leaf) !important; border:0 !important; }
[data-testid="stFileUploader"] section button *{ color:#fff !important; }
.footer{ color:var(--muted); font-size:.82rem; margin-top:2.2rem; padding-top:1rem; border-top:1px solid var(--line); line-height:1.5; }

@media (max-width:900px){
  .hero{ grid-template-columns:1fr; padding:1.5rem 1.3rem; gap:1.2rem; }
}
@media (max-width:640px){
  .block-container{ padding-left:.9rem !important; padding-right:.9rem !important; }
  .res-top{ padding:1.2rem; } .res-body{ padding:1rem; }
  .gauge{ width:88px; height:88px; } .gauge svg{ width:88px; height:88px; }
  .rank{ grid-template-columns:1.6rem 1.3rem minmax(0,5rem) 1fr 2.8rem; gap:.35rem; padding:.45rem .3rem; }
  .brand-sub{ display:none; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    model = joblib.load(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    return model, encoder

def tidy(text):
    return re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip()


SOIL_FIXES = {"light soi": "light soil"}


def build_options(raw_values):
    seen = {}
    for raw in raw_values:
        key = SOIL_FIXES.get(tidy(raw).lower(), tidy(raw).lower())
        if key not in seen or str(raw) == tidy(raw):
            seen[key] = raw
    return sorted([(k[:1].upper() + k[1:], v) for k, v in seen.items()], key=lambda t: t[0].lower())


if not (MODEL_FILE.exists() and ENCODER_FILE.exists()):
    st.markdown(H('<div class="hero"><div><h1>Model <em>not found</em></h1>'
                  '<p>Run the notebook first so it can save the trained model.</p></div></div>'),
                unsafe_allow_html=True)
    st.error("Missing `best_crop_model.pkl` and/or `crop_label_encoder.pkl`.")
    st.markdown("Place both files in the **same folder as `app.py`**, then refresh this page.")
    st.stop()

model, encoder = load_artifacts()
pre = model.named_steps["preprocessor"]
numeric_cols, categorical_cols, cat_categories = [], [], {}
for _name, _tr, _cols in pre.transformers_:
    if _name == "num":
        numeric_cols = list(_cols)
    elif _name == "cat":
        categorical_cols = list(_cols)
        for _c, _cats in zip(_cols, _tr.categories_):
            cat_categories[_c] = list(_cats)
all_features = numeric_cols + categorical_cols
n_crops = len(encoder.classes_)
final_estimator = type(model.named_steps["model"]).__name__
model_short = final_estimator.replace("Classifier", "")
soil_opts = build_options(cat_categories["SOIL"])
soil_label_of = {raw: label for label, raw in soil_opts}


def sort_months(vals):
    return sorted(vals, key=lambda m: MONTHS.index(m) if m in MONTHS else 99)


def canon_map(col):
    m = {}
    for c in cat_categories[col]:
        m.setdefault(SOIL_FIXES.get(tidy(c).lower(), tidy(c).lower()), c)
    for c in cat_categories[col]:
        if str(c) == tidy(c):
            m[SOIL_FIXES.get(tidy(c).lower(), tidy(c).lower())] = c
    return m


def predict_proba(df):
    return model.predict_proba(df[all_features])


# ----------------------------------------------------------------------------- state
def default_month(col, wanted):
    return wanted if wanted in cat_categories[col] else cat_categories[col][0]


def init_state():
    ss = st.session_state
    ss.setdefault("page", "Recommend")
    ss.setdefault("result", None)
    ss.setdefault("history", [])
    ss.setdefault("batch", None)
    ss.setdefault("feedback_status", None)
    for k, (_l, _u, _lo, _hi, v, _s) in NUMERIC_SPEC.items():
        ss.setdefault(f"in_{k}", v)
    ss.setdefault("in_SOIL", soil_opts[0][1])
    ss.setdefault("in_WATER_SOURCE", cat_categories["WATER_SOURCE"][0])
    ss.setdefault("in_SEASON", cat_categories["SEASON"][0])
    ss.setdefault("in_SOWN", default_month("SOWN", "Jun"))
    ss.setdefault("in_HARVESTED", default_month("HARVESTED", "Sep"))


def go(page):
    st.session_state.page = page


def load_preset(name):
    for key, val in PRESETS[name].items():
        if key in ("SOIL", "WATER_SOURCE", "SEASON"):
            val = canon_map(key).get(tidy(val).lower(), val)
        st.session_state[f"in_{key}"] = val


def reset_inputs():
    for k, (_l, _u, _lo, _hi, v, _s) in NUMERIC_SPEC.items():
        st.session_state[f"in_{k}"] = v
    st.session_state.in_SOIL = soil_opts[0][1]
    st.session_state.in_WATER_SOURCE = cat_categories["WATER_SOURCE"][0]
    st.session_state.in_SEASON = cat_categories["SEASON"][0]
    st.session_state.in_SOWN = default_month("SOWN", "Jun")
    st.session_state.in_HARVESTED = default_month("HARVESTED", "Sep")
    st.session_state.result = None


init_state()

# ----------------------------------------------------------------------------- UI helpers
LOGO = ('<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#D6EB7A" stroke-width="2" '
        'stroke-linecap="round" stroke-linejoin="round"><path d="M12 22V9"/><path d="M12 9c0-3 2-5 5-5 0 3-2 5-5 5z"/>'
        '<path d="M12 13c0-3-2-5-5-5 0 3 2 5 5 5z"/><path d="M5 22h14"/></svg>')


def render_nav():
    with st.container(key="navbar"):
        c0, *cols = st.columns([1.7, 1, 1, 1, 1, 1], vertical_alignment="center")
        with c0:
            st.markdown(H(f'<div class="brand"><div class="brand-mark">{LOGO}</div><div>'
                          f'<div class="brand-name">Crop Studio</div><div class="brand-sub">AI crop advisor</div>'
                          f'</div></div>'), unsafe_allow_html=True)
        for col, p in zip(cols, PAGES):
            with col:
                st.button(p, key=f"nav_{p}", **S_BTN, on_click=go, args=(p,),
                          type="primary" if st.session_state.page == p else "secondary")


def page_head(title, sub):
    st.markdown(f'<div class="page-title">{title}</div><p class="page-sub">{sub}</p>', unsafe_allow_html=True)


def sec_head(icon, title, hint):
    st.markdown(H(f'<div class="sec-head"><div class="sec-ic">{icon}</div><div><h3>{title}</h3><p>{hint}</p></div></div>'),
                unsafe_allow_html=True)


def kpi_row(items):
    for col, (v, l) in zip(st.columns(len(items)), items):
        with col:
            st.markdown(H(f'<div class="kpi"><b>{v}</b><span>{l}</span></div>'), unsafe_allow_html=True)


def stepper(steps, active):
    out = '<div class="stepper">'
    for i, (t, d) in enumerate(steps):
        cls = "s-done" if i < active else ("s-active" if i == active else "")
        mark = "✓" if i < active else str(i + 1)
        out += f'<div class="step {cls}"><div class="circle">{mark}</div><div class="step-title">{t}</div><div class="step-desc">{d}</div></div>'
        if i < len(steps) - 1:
            out += f'<div class="line {"done" if i < active else ""}"></div>'
    st.markdown(H(out + "</div>"), unsafe_allow_html=True)


def copy_button(text, label="Copy result"):
    payload = json.dumps(text).replace("</", "<\\/")
    page = f"""<style>body{{margin:0}}</style>
    <button id="b" style="font:600 14px Inter,Arial,sans-serif;color:#14241B;background:#fff;border:1px solid #E0E6D9;
      border-radius:12px;padding:10px 16px;cursor:pointer;width:100%;height:42px">{label}</button>
    <script>
      const t={payload}, b=document.getElementById('b'), l=b.textContent;
      b.onclick=async()=>{{let ok=false;try{{await navigator.clipboard.writeText(t);ok=true}}catch(e){{
        const a=document.createElement('textarea');a.value=t;document.body.appendChild(a);a.select();
        try{{ok=document.execCommand('copy')}}catch(e){{}}document.body.removeChild(a)}}
        b.textContent=ok?'Copied ✓':'Copy blocked by browser';setTimeout(()=>b.textContent=l,2000)}};
    </script>"""
    if hasattr(st, "iframe"):
        st.iframe(page, height=46)
    else:
        components.html(page, height=46)


def gauge_svg(pct):
    r, c = 48, 2 * 3.14159265 * 48
    dash = c * max(min(pct, 100), 0) / 100
    return (f'<div class="gauge"><svg width="112" height="112" viewBox="0 0 112 112">'
            f'<circle cx="56" cy="56" r="{r}" fill="none" stroke="rgba(255,255,255,.18)" stroke-width="9"/>'
            f'<circle cx="56" cy="56" r="{r}" fill="none" stroke="#D6EB7A" stroke-width="9" stroke-linecap="round" '
            f'stroke-dasharray="{dash:.1f} {c:.1f}"/></svg>'
            f'<div class="g-val"><b>{pct:.0f}%</b><span>confidence</span></div></div>')


def result_card(res):
    top = res["top"]
    best, best_p = top[0]
    rows = ""
    for i, (n, p) in enumerate(top):
        rows += (f'<div class="rank {"first" if i == 0 else ""}"><span class="n">{i + 1}</span>'
                 f'<span class="e">{emoji_for(n)}</span><span class="name" title="{n}">{n}</span>'
                 f'<div class="track"><div class="fill" style="width:{max(p * 100, 1.5):.1f}%"></div></div>'
                 f'<span class="pct">{p * 100:.1f}%</span></div>')
    v = res["values"]
    margin = (top[0][1] - top[1][1]) * 100
    why = (f'<div class="why"><b>Why this result?</b> The model compared your field — '
           f'<b>{soil_label_of.get(v["SOIL"], tidy(v["SOIL"]))}</b>, pH <b>{v["SOIL_PH"]:.1f}</b>, '
           f'<b>{v["TEMP"]:.0f}°C</b>, <b>{v["RELATIVE_HUMIDITY"]:.0f}%</b> humidity, '
           f'N-P-K <b>{v["N"]:.0f}-{v["P"]:.0f}-{v["K"]:.0f}</b> in the '
           f'<b>{SEASON_LABEL.get(str(v["SEASON"]).lower(), str(v["SEASON"]))}</b> season — '
           f'against {n_crops} crops learned from historical records. '
           f'<b>{best.title()}</b> leads the runner-up by <b>{margin:.1f}</b> percentage points.</div>')
    warn = ""
    if best_p < 0.5:
        warn = ('<div class="warn"><b>Heads up:</b> confidence is below 50%, so these conditions suit several crops '
                'almost equally. Treat the list as a shortlist, not a single answer.</div>')
    chips = "".join(f'<span class="chip"><b>{k}</b>{val}</span>' for k, val in res["inputs"].items())
    st.markdown(H(f"""
    <div class="res">
      <div class="res-top">
        <div><div class="tag">Best match for your field</div>
          <div class="crop"><span class="em">{emoji_for(best)}</span>{best}</div>
          <div class="sub">Top pick out of {n_crops} crops</div></div>
        {gauge_svg(best_p * 100)}
      </div>
      <div class="res-body"><h4>Top 5 recommendations</h4>{rows}{why}{warn}<div class="chips">{chips}</div></div>
    </div>"""), unsafe_allow_html=True)


def snapshot(values):
    tiles = ""
    for k in ["SOIL_PH", "TEMP", "RELATIVE_HUMIDITY", "WATERREQUIRED", "N", "P", "K"]:
        l, u, lo, hi, _d, _s = NUMERIC_SPEC[k]
        val = float(values[k])
        pct = max(0, min(100, (val - lo) / (hi - lo) * 100))
        shown = f"{val:.1f}" if k == "SOIL_PH" else f"{val:.0f}"
        tiles += (f'<div class="tile"><div class="k">{l}</div><div class="v">{shown}<small>{u}</small></div>'
                  f'<div class="mini"><i style="width:{pct:.0f}%"></i></div></div>')
    st.markdown(H(f'<div class="snap-title">Field snapshot</div><div class="snap">{tiles}</div>'), unsafe_allow_html=True)


def tidy_inputs(values, soil_raw):
    return {
        "Soil": soil_label_of.get(soil_raw, tidy(soil_raw)),
        "pH": f'{values["SOIL_PH"]:.1f}', "Temp": f'{values["TEMP"]:.1f}°C',
        "Humidity": f'{values["RELATIVE_HUMIDITY"]:.0f}%',
        "N-P-K": f'{values["N"]:.0f}-{values["P"]:.0f}-{values["K"]:.0f}',
        "Season": SEASON_LABEL.get(str(values["SEASON"]).lower(), str(values["SEASON"])),
        "Water": WATER_LABEL.get(str(values["WATER_SOURCE"]).lower(), str(values["WATER_SOURCE"])),
        "Sow→Harvest": f'{values["SOWN"]}→{values["HARVESTED"]}',
    }


# ----------------------------------------------------------------------------- pages
def page_recommend():
    stats = [("🌱", n_crops, "Crops covered"), ("🧪", len(all_features), "Field factors"),
             ("🤖", model_short, "Prediction model"), ("📊", f"{DATASET_ROWS // 1000}K", "Training records")]
    stat_html = "".join(f'<div class="stat"><div class="ic">{i}</div><b>{v}</b><span>{l}</span></div>' for i, v, l in stats)
    st.markdown(H(f"""
    <div class="hero"><div>
      <span class="pill"><span class="dot"></span>AI-powered agriculture</span>
      <h1>Find the best crop for <em>your field</em></h1>
      <p>Enter soil, climate and nutrient conditions. The model ranks the crops most likely to thrive — with a confidence score and close alternatives.</p>
    </div><div class="stats">{stat_html}</div></div>"""), unsafe_allow_html=True)

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        st.markdown('<div class="lbl">Quick start — load an example field</div>', unsafe_allow_html=True)
        pc = st.columns(len(PRESETS) + 1)
        for col, name in zip(pc, PRESETS):
            with col:
                st.button(name, key=f"preset_{name}", **S_BTN, on_click=load_preset, args=(name,))
        with pc[-1]:
            st.button("↺ Reset", key="reset_inputs", **S_BTN, on_click=reset_inputs)
        st.write("")

        with st.form("crop_form", border=False):
            with st.container(border=True, key="sec_soil"):
                sec_head("🪨", "Soil & water", "What the field is made of and how it is watered")
                c1, c2 = st.columns(2)
                with c1:
                    st.selectbox("Soil type", [r for _l, r in soil_opts], key="in_SOIL",
                                 format_func=lambda r: soil_label_of.get(r, tidy(r)))
                with c2:
                    st.selectbox("Water source", cat_categories["WATER_SOURCE"], key="in_WATER_SOURCE",
                                 format_func=lambda v: WATER_LABEL.get(str(v).lower(), tidy(v).title()))
                c1, c2 = st.columns(2)
                for col, k in zip((c1, c2), ("SOIL_PH", "WATERREQUIRED")):
                    l, u, lo, hi, _v, stp = NUMERIC_SPEC[k]
                    with col:
                        st.slider(f"{l} ({u})" if u else l, lo, hi, step=stp, key=f"in_{k}")
            with st.container(border=True, key="sec_climate"):
                sec_head("🌤️", "Climate", "Typical conditions during the growing period")
                c1, c2 = st.columns(2)
                for col, k in zip((c1, c2), ("TEMP", "RELATIVE_HUMIDITY")):
                    l, u, lo, hi, _v, stp = NUMERIC_SPEC[k]
                    with col:
                        st.slider(f"{l} ({u})", lo, hi, step=stp, key=f"in_{k}")
            with st.container(border=True, key="sec_nutrients"):
                sec_head("🧪", "Soil nutrients", "Available nitrogen, phosphorus and potassium (kg/ha)")
                for col, k in zip(st.columns(3), ("N", "P", "K")):
                    l, u, lo, hi, _v, stp = NUMERIC_SPEC[k]
                    with col:
                        st.slider(l, lo, hi, step=stp, key=f"in_{k}")
            with st.container(border=True, key="sec_calendar"):
                sec_head("📅", "Crop calendar", "When you plan to sow and harvest")
                c1, c2 = st.columns(2)
                with c1:
                    st.selectbox("Season", cat_categories["SEASON"], key="in_SEASON",
                                 format_func=lambda v: SEASON_LABEL.get(str(v).lower(), tidy(v).title()))
                with c2:
                    l, u, lo, hi, _v, stp = NUMERIC_SPEC["CROPDURATION"]
                    st.slider(f"{l} ({u})", lo, hi, step=stp, key="in_CROPDURATION")
                c1, c2 = st.columns(2)
                with c1:
                    st.selectbox("Sowing month", sort_months(cat_categories["SOWN"]), key="in_SOWN",
                                 format_func=lambda m: MONTH_FULL.get(m, m))
                with c2:
                    st.selectbox("Harvest month", sort_months(cat_categories["HARVESTED"]), key="in_HARVESTED",
                                 format_func=lambda m: MONTH_FULL.get(m, m))
            submitted = st.form_submit_button("🌾  Recommend a crop", type="primary", **S_SUB)

    if submitted:
        ss = st.session_state
        if ss.in_SOWN == ss.in_HARVESTED:
            st.warning("Sowing and harvest month are the same — please check your crop calendar.")
        with st.spinner("Analysing field conditions…"):
            values = {k: ss[f"in_{k}"] for k in all_features}
            probs = predict_proba(pd.DataFrame([values]))[0]
            order = np.argsort(probs)[::-1][:5]
            top = [(str(encoder.classes_[i]), float(probs[i])) for i in order]
            inputs = tidy_inputs(values, values["SOIL"])
            ss.result = {"top": top, "inputs": inputs, "values": values}
            ss.history.insert(0, {"Time": datetime.now().strftime("%H:%M:%S"), "Best crop": top[0][0],
                                  "Confidence %": round(top[0][1] * 100, 1),
                                  "2nd": top[1][0], "3rd": top[2][0], **inputs})

    with right:
        res = st.session_state.result
        if not res:
            st.markdown(H('<div class="empty"><div class="big">🌱</div><h3>Your recommendation will appear here</h3>'
                          '<p>Set the field conditions or load an example, then press <b>Recommend a crop</b>. '
                          'You will get the best match plus the four closest alternatives.</p></div>'),
                        unsafe_allow_html=True)
        else:
            result_card(res)
            best, bp = res["top"][0]
            st.write("")
            c1, c2, c3 = st.columns(3)
            with c1:
                copy_button(f"Recommended crop: {best} ({bp * 100:.1f}% confidence). Alternatives: "
                            + ", ".join(f"{n} ({p * 100:.1f}%)" for n, p in res["top"][1:]))
            with c2:
                st.download_button("⬇ JSON", json.dumps(
                    {"recommendation": best, "confidence": round(bp, 4),
                     "top5": [{"crop": n, "probability": round(p, 4)} for n, p in res["top"]],
                     "inputs": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in res["values"].items()}},
                    indent=2), "crop_recommendation.json", "application/json", **S_DL)
            with c3:
                st.button("History →", key="to_hist", **S_BTN, on_click=go, args=("History",))

    if st.session_state.result:
        snapshot(st.session_state.result["values"])


def page_batch():
    page_head("Batch recommendations",
              "Upload a CSV with one row per field and get a recommendation for every row, ready to download.")
    steps = [("Upload CSV", "Add your field data"), ("Validate", "Check columns and values"),
             ("Recommend", "Predict a crop per row")]
    b = st.session_state.batch
    stepper(steps, 0 if not b else (3 if b.get("ok") else 1))

    template = pd.DataFrame([{**{k: NUMERIC_SPEC[k][4] for k in numeric_cols},
                              "SOIL": soil_opts[0][1], "SEASON": cat_categories["SEASON"][0],
                              "SOWN": default_month("SOWN", "Jun"), "HARVESTED": default_month("HARVESTED", "Sep"),
                              "WATER_SOURCE": cat_categories["WATER_SOURCE"][0]}])[all_features]
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    with c1:
        up = st.file_uploader("Field data (CSV)", type=["csv"], key="batch_up")
    with c2:
        st.download_button("⬇ Download template", template.to_csv(index=False), "field_template.csv", "text/csv", **S_DL)
    st.caption("Required columns: " + ", ".join(f"`{c}`" for c in all_features) + f" · up to {MAX_MB} MB")

    if up is not None:
        data = up.getvalue()
        sig = (up.name, len(data))
        if not b or b.get("sig") != sig:
            with st.spinner("Running predictions…"):
                st.session_state.batch = b = run_batch(up.name, data, sig)
            st.rerun()
    if not b:
        st.markdown(H('<div class="empty"><div class="big">📄</div><h3>No file uploaded yet</h3>'
                      '<p>Download the template, fill in your fields, and upload it above.</p></div>'),
                    unsafe_allow_html=True)
        return
    if not b["ok"]:
        st.error(b["error"])
        return
    for w in b["warnings"]:
        st.warning(w)
    df = b["df"]
    kpi_row([(len(df), "rows predicted"), (df["Best crop"].nunique(), "different crops"),
             (f'{df["Confidence %"].mean():.1f}%', "average confidence"),
             (int((df["Confidence %"] < 50).sum()), "low-confidence rows")])
    st.write("")
    left, right = st.columns([1.6, 1], gap="large")
    with left:
        st.markdown("**Results**")
        st.dataframe(df, height=360, hide_index=True, **S_DF)
        st.download_button("⬇ Download results (CSV)", df.to_csv(index=False), "crop_recommendations.csv", "text/csv")
    with right:
        st.markdown("**Recommended crops**")
        st.bar_chart(df["Best crop"].value_counts().head(12), color="#2F7044", horizontal=True)


def run_batch(name, data, sig):
    out = {"sig": sig, "ok": False, "error": "", "warnings": [], "df": None}
    if len(data) > MAX_MB * 1024 * 1024:
        out["error"] = f"File exceeds the {MAX_MB} MB limit."
        return out
    try:
        raw = pd.read_csv(io.BytesIO(data))
    except Exception as e:
        out["error"] = f"Could not read this file as CSV: {e}"
        return out
    raw.columns = [re.sub(r"\s+", "_", str(c).strip()).upper() for c in raw.columns]
    missing = [c for c in all_features if c not in raw.columns]
    if missing:
        out["error"] = "Missing required column(s): " + ", ".join(missing) + ". Download the template for the exact layout."
        return out
    df = raw[all_features].copy()
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = df[numeric_cols].isna().any(axis=1)
    if bad.any():
        out["warnings"].append(f"{int(bad.sum())} row(s) with missing or non-numeric values were skipped (rows: "
                               + ", ".join(str(i + 2) for i in df.index[bad][:8]) + (" …" if bad.sum() > 8 else "") + ").")
    for c in categorical_cols:
        cmap = canon_map(c)
        mapped = df[c].map(lambda v: cmap.get(SOIL_FIXES.get(tidy(v).lower(), tidy(v).lower())))
        unseen = mapped.isna() & ~bad
        if unseen.any():
            vals = ", ".join(sorted({tidy(v) for v in df.loc[unseen, c]})[:5])
            out["warnings"].append(f"{int(unseen.sum())} row(s) have a {c} value the model has not seen ({vals}); "
                                   "predictions for those rows are less reliable.")
        df[c] = mapped.where(mapped.notna(), df[c])
    ok = df[~bad]
    if ok.empty:
        out["error"] = "No valid rows were found in this file."
        return out
    probs = predict_proba(ok)
    order = np.argsort(probs, axis=1)[:, ::-1][:, :3]
    classes = np.array([str(c) for c in encoder.classes_])
    res = raw.loc[ok.index, [c for c in raw.columns if c in all_features]].copy()
    res.insert(0, "Row", ok.index + 2)
    res.insert(1, "Best crop", classes[order[:, 0]])
    res.insert(2, "Confidence %", np.round(probs[np.arange(len(ok)), order[:, 0]] * 100, 1))
    res.insert(3, "2nd choice", classes[order[:, 1]])
    res.insert(4, "3rd choice", classes[order[:, 2]])
    out.update(ok=True, df=res.reset_index(drop=True))
    return out


def page_history():
    page_head("Prediction history", "Every recommendation from this session. History is cleared when the page is reloaded.")
    h = st.session_state.history
    if not h:
        st.markdown(H('<div class="empty"><div class="big">🕘</div><h3>Nothing here yet</h3>'
                      '<p>Run a recommendation and it will be listed here.</p></div>'), unsafe_allow_html=True)
        st.write("")
        st.button("Go to Recommend", on_click=go, args=("Recommend",), type="primary")
        return
    df = pd.DataFrame(h)
    kpi_row([(len(df), "predictions"), (df["Best crop"].mode()[0], "most recommended"),
             (f'{df["Confidence %"].mean():.1f}%', "average confidence")])
    st.write("")
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        st.dataframe(df, hide_index=True, **S_DF)
        c1, c2, _ = st.columns([1, 1, 2])
        with c1:
            st.download_button("⬇ Download CSV", df.to_csv(index=False), "prediction_history.csv", "text/csv", **S_DL)
        with c2:
            st.button("Clear history", on_click=lambda: st.session_state.update(history=[], result=None), **S_BTN)
    with right:
        st.markdown("**Crops recommended**")
        st.bar_chart(df["Best crop"].value_counts(), color="#2F7044", horizontal=True)


def page_about():
    page_head("About this tool", "How the recommendation is produced, what it uses, and what it cannot do.")
    st.markdown("**How it works**")
    stepper([("Describe the field", "Soil, climate, nutrients, calendar"),
             ("Model scoring", f"{model_short} scores every crop"),
             ("Ranked answer", "Best match plus four alternatives")], -1)
    st.markdown("**Features**")
    feats = [("⚡", "Instant recommendation", "Best crop with a confidence score and the closest alternatives."),
             ("🧭", "Example fields", "Load a typical set-up in one click, then adjust it."),
             ("📦", "Batch mode", "Upload a CSV of many fields and download a recommendation for each."),
             ("🗂️", "History & export", "Keep a session log; export results as CSV or JSON."),
             ("✅", "Data checks", "Missing columns, bad numbers and unseen categories are reported, not hidden."),
             ("🎯", "Honest confidence", "A warning appears whenever the model cannot separate several crops.")]
    cols = st.columns(3)
    for i, (ic, t, d) in enumerate(feats):
        with cols[i % 3]:
            st.markdown(H(f'<div class="card"><div class="ic">{ic}</div><h4>{t}</h4><p>{d}</p></div>'), unsafe_allow_html=True)
    st.markdown("**Model**")
    kpi_row([(model_short, "selected model"), (n_crops, "crops"), (len(numeric_cols), "numeric inputs"),
             (len(categorical_cols), "categorical inputs")])
    st.write("")
    with st.expander(f"Crops the model can recommend ({n_crops})"):
        st.markdown("".join(f'<span class="crop-chip">{emoji_for(c)} {c}</span>' for c in sorted(str(c) for c in encoder.classes_)),
                    unsafe_allow_html=True)
    st.markdown(H('<div class="warn">Recommendations come from patterns in historical crop records. They support planning '
                  'and do not replace local agronomic advice. The model only knows the crops listed above.</div>'),
                unsafe_allow_html=True)


def page_feedback():
    page_head("Feedback", "Tell us what works and what does not. Responses are saved locally on this machine.")
    with st.container(key="feedback_wrap"):
        with st.form("feedback_form", clear_on_submit=True, border=True):
            name = st.text_input("Name (optional)", placeholder="Your name")
            cat = st.selectbox("Category", ["General", "Bug report", "Feature request", "Design", "Other"])
            rating = st.slider("Your experience", 1, 5, 4)
            msg = st.text_area("Your feedback", placeholder="Tell us what you think…")
            sent = st.form_submit_button("Submit feedback", type="primary")
    if sent:
        if not msg.strip():
            st.session_state.feedback_status = ("error", "Please write a message before submitting.")
        else:
            entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), "name": name.strip(),
                     "category": cat, "rating": rating, "message": msg.strip()}
            try:
                data = json.loads(FEEDBACK_FILE.read_text("utf-8")) if FEEDBACK_FILE.exists() else []
                data.append(entry)
                FEEDBACK_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
                st.session_state.feedback_status = ("success", f"Saved to {FEEDBACK_FILE.name}. Thank you!")
            except Exception as e:
                st.session_state.feedback_status = ("error", f"Could not save feedback. Check that the app folder is writable. Details: {e}")
    s = st.session_state.feedback_status
    if s:
        (st.success if s[0] == "success" else st.error)(s[1])


# ----------------------------------------------------------------------------- main
render_nav()
{"Recommend": page_recommend, "Batch": page_batch, "History": page_history,
 "About": page_about, "Feedback": page_feedback}[st.session_state.page]()
st.markdown(f'<div class="footer">Predictions come from a {final_estimator} model trained on historical crop records. '
            'They support planning decisions and do not replace local agronomic advice.</div>', unsafe_allow_html=True)