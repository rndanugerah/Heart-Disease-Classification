import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
import plotly.express as px

# Konfigurasi Halaman
st.set_page_config(
    page_title="CardioSense AI | Multi-Model Heart Predictor",
    page_icon="❤️",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #ffffff; }
    .stButton>button {
        width: 100%; border-radius: 10px; height: 3em;
        background-color: #ff4b4b; color: white; font-weight: bold;
    }
    .css-1r6slb0 { background: rgba(255, 255, 255, 0.05); border-radius: 15px; padding: 20px; border: 1px solid rgba(255, 255, 255, 0.1); }
    h1 { color: #ff4b4b; text-align: center; }
    .accuracy-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 15px; border-radius: 10px; text-align: center; margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Data Akurasi dari Notebook (Mapping)
MODEL_ACCURACIES = {
    "LOGISTIC REGRESSION": 80.00,
    "KNN OPTIMIZED": 78.33,
    "SVM GRIDSEARCH": 83.33,
    "NAIVE BAYES": 81.67,
    "DECISION TREE": 73.33,
    "RANDOM FOREST": 75.00,
    "XGBOOST": 80.00
}

# Fungsi untuk mendeteksi model yang tersedia di folder 'model'
def get_available_models():
    model_dir = 'model'
    if not os.path.exists(model_dir):
        return {}
    
    files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
    models = {}
    for f in files:
        display_name = f.replace('heart_disease_', '').replace('.pkl', '').replace('_', ' ').upper()
        models[display_name] = os.path.join(model_dir, f)
    return models

available_models = get_available_models()

# Load Scaler
@st.cache_resource
def load_scaler():
    paths = ['scaler.pkl', 'model/scaler.pkl']
    for path in paths:
        if os.path.exists(path):
            return joblib.load(path)
    return None

scaler = load_scaler()

# Load Selected Model
@st.cache_resource
def load_model(file_path):
    return joblib.load(file_path)

# UI Sidebar
st.sidebar.title("⚙️ Model Settings")
if available_models:
    selected_model_name = st.sidebar.selectbox("Pilih Model Machine Learning", list(available_models.keys()))
    model_file = available_models[selected_model_name]
    model = load_model(model_file)
    
    # Tampilkan Akurasi di Sidebar
    accuracy = MODEL_ACCURACIES.get(selected_model_name, "N/A")
    st.sidebar.markdown(f"""
        <div class="accuracy-card">
            <small style="color:white;">Akurasi Model Training</small>
            <h2 style="margin:0; color:white; ">{accuracy}%</h2>
        </div>
    """, unsafe_allow_html=True)

    # Pie Chart Perbandingan (Berdasarkan Gambar User)
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Perbandingan Akurasi")
    
    chart_data = pd.DataFrame({
        "Model": ["Logistic Regression", "KNN V2", "SVM V2", "Naive Bayes", "Decision Tree", "Random Forest", "XGBoost"],
        "Akurasi": [74, 82, 75, 77, 69, 80, 79]
    })
    
    fig = px.pie(chart_data, values='Akurasi', names='Model', 
                 hole=0.4,
                 color_discrete_sequence=px.colors.qualitative.Pastel)
    
    fig.update_layout(
        height=250, # Mengatur tinggi agar lebih rapat ke judul
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white")
    )
    fig.update_traces(
        textposition='inside', 
        textinfo='label+value',
        texttemplate='%{label}<br>%{value}%',
        hovertemplate="<b>%{label}</b><br>Akurasi: %{value}%<extra></extra>"
    )
    
    st.sidebar.plotly_chart(fig, use_container_width=True)
else:
    st.sidebar.error("Model tidak ditemukan di folder 'model/'")
    st.stop()

# UI Header
st.title("❤️ CardioSense AI")
st.markdown(f"<p style='text-align: center;'>Analisis Risiko Jantung menggunakan Model <b>{selected_model_name}</b></p>", unsafe_allow_html=True)
st.divider()

# Layout Input (Baris Padat)
with st.container():
    row1 = st.columns(5)
    with row1[0]: age = st.number_input("Umur", 1, 120, 45)
    with row1[1]: sex = st.selectbox("Gender", options=[(0, "Wanita"), (1, "Pria")], format_func=lambda x: x[1])[0]
    with row1[2]: cp = st.selectbox("Tipe Nyeri", options=[(1, "Typical"), (2, "Atypical"), (3, "Non-ang"), (4, "Asym")], format_func=lambda x: x[1])[0]
    with row1[3]: trestbps = st.number_input("Tekanan Darah", 80, 250, 120)
    with row1[4]: chol = st.number_input("Kolesterol", 100, 600, 200)

    row2 = st.columns(4)
    with row2[0]: fbs = st.selectbox("Gula Darah > 120", options=[(0, "Tidak"), (1, "Ya")], format_func=lambda x: x[1])[0]
    with row2[1]: restecg = st.selectbox("EKG Istirahat", options=[(0, "Normal"), (1, "ST-T"), (2, "Hipe")], format_func=lambda x: x[1])[0]
    with row2[2]: thalach = st.number_input("Detak Jantung Max", 60, 220, 150)
    with row2[3]: exang = st.selectbox("Angina Olahraga", options=[(0, "Tidak"), (1, "Ya")], format_func=lambda x: x[1])[0]

    row3 = st.columns(4)
    with row3[0]: oldpeak = st.number_input("Depresi ST", 0.0, 10.0, 0.0)
    with row3[1]: slope = st.selectbox("Slope", options=[(1, "Up"), (2, "Flat"), (3, "Down")], format_func=lambda x: x[1])[0]
    with row3[2]: ca = st.selectbox("Pembuluh Utama", options=[0, 1, 2, 3])
    with row3[3]: thal = st.selectbox("Thalassemia", options=[(3, "Normal"), (6, "Fixed"), (7, "Revers")], format_func=lambda x: x[1])[0]

def get_prediction(processed_df):
    column_order = [
        'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 
        'sex_1.0', 'cp_2.0', 'cp_3.0', 'cp_4.0', 'fbs_1.0', 
        'restecg_1.0', 'restecg_2.0', 'exang_1.0', 'slope_2.0', 
        'slope_3.0', 'ca_1.0', 'ca_2.0', 'ca_3.0', 'thal_6.0', 'thal_7.0'
    ]
    processed_df = processed_df[column_order]
    input_scaled = scaler.transform(processed_df)
    prediction = model.predict(input_scaled)
    
    probability = None
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_scaled)[0][1]
    elif hasattr(model, "decision_function"):
        df_val = model.decision_function(input_scaled)
        val = df_val[0] if isinstance(df_val, np.ndarray) else df_val
        probability = 1 / (1 + np.exp(-val))
    
    return prediction[0], probability

st.divider()

if st.button("ANALISIS SEKARANG"):
    features = {
        'age': age, 'trestbps': trestbps, 'chol': chol, 'thalach': thalach, 'oldpeak': oldpeak,
        'sex_1.0': 1 if sex == 1 else 0,
        'cp_2.0': 1 if cp == 2 else 0, 'cp_3.0': 1 if cp == 3 else 0, 'cp_4.0': 1 if cp == 4 else 0,
        'fbs_1.0': 1 if fbs == 1 else 0,
        'restecg_1.0': 1 if restecg == 1 else 0, 'restecg_2.0': 1 if restecg == 2 else 0,
        'exang_1.0': 1 if exang == 1 else 0,
        'slope_2.0': 1 if slope == 2 else 0, 'slope_3.0': 1 if slope == 3 else 0,
        'ca_1.0': 1 if ca == 1 else 0, 'ca_2.0': 1 if ca == 2 else 0, 'ca_3.0': 1 if ca == 3 else 0,
        'thal_6.0': 1 if thal == 6 else 0, 'thal_7.0': 1 if thal == 7 else 0,
    }
    
    input_df = pd.DataFrame([features])
    
    if scaler is not None:
        with st.spinner(f'Menganalisis...'):
            result, prob = get_prediction(input_df)
            
        st.subheader(f"Hasil Analisis ({selected_model_name})")
        if result == 1:
            st.error("⚠️ HASIL: RISIKO TINGGI")
        else:
            st.success("✅ HASIL: RISIKO RENDAH")
        
        if prob is not None:
            st.write(f"**Confidence Level:** {prob*100:.2f}%")
            st.progress(prob)
    else:
        st.error("Scaler tidak ditemukan.")

st.markdown("---")
st.caption(f"Multi-Model CardioSense | Data Accuracy based on Notebook Training")