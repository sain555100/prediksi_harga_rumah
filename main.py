
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Title of the Streamlit app
st.title('Aplikasi Prediksi Harga Rumah')
st.write('Masukkan detail rumah untuk memprediksi harganya.')

# Load saved artifacts
try:
    with open('rentang_fitur.pkl', 'rb') as file:
        rentang_fitur = pickle.load(file)
    with open('label_encoder.pkl', 'rb') as file:
        encoders = pickle.load(file)
    with open('scaler_fitur.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('target_scaler.pkl', 'rb') as file:
        target_scaler = pickle.load(file)
    # The model was saved as rf_model but named xgboost_model.pkl
    with open('xgboost_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Error: Pastikan semua file model (rentang_fitur.pkl, label_encoder.pkl, scaler_fitur.pkl, target_scaler.pkl, gb_model.pkl) tersedia di direktori yang sama dengan main.py.")
    st.stop()


# Helper function to validate numerical inputs based on min/max ranges
def validasi_numerik(nilai, fitur):
    min_val = rentang_fitur[fitur]['min']
    max_val = rentang_fitur[fitur]['max']
    # Streamlit sliders already enforce min/max, but this is a safeguard
    return max(min_val, min(nilai, max_val))

# Input fields for numerical features (using sliders)
st.subheader('Input Fitur Numerik')

# Bedrooms - ensure it's an integer for the slider, check rentang_fitur
bedrooms_min = int(rentang_fitur['bedrooms']['min'])
bedrooms_max = int(rentang_fitur['bedrooms']['max'])
in_bed = st.slider('Jumlah Kamar Tidur (Bedrooms)', bedrooms_min, bedrooms_max, 3)

# Bathrooms - float slider
bathrooms_min = float(rentang_fitur['bathrooms']['min'])
bathrooms_max = float(rentang_fitur['bathrooms']['max'])
# Step can be 0.25 as bathrooms often come in .25, .5, .75 increments
in_bath = st.slider('Jumlah Kamar Mandi (Bathrooms)', bathrooms_min, bathrooms_max, 2.0, step=0.25)

# Sqft Living
sqft_living_min = int(rentang_fitur['sqft_living']['min'])
sqft_living_max = int(rentang_fitur['sqft_living']['max'])
in_sqft = st.slider('Luas Ruang Tamu (sqft_living)', sqft_living_min, sqft_living_max, 2000)

# Floors
floors_min = float(rentang_fitur['floors']['min'])
floors_max = float(rentang_fitur['floors']['max'])
in_flr = st.slider('Jumlah Lantai (Floors)', floors_min, floors_max, 1.5, step=0.5)

# Sqft Above
sqft_above_min = int(rentang_fitur['sqft_above']['min'])
sqft_above_max = int(rentang_fitur['sqft_above']['max'])
in_abv = st.slider('Luas di Atas Tanah (sqft_above)', sqft_above_min, sqft_above_max, 1500)

# Validate numerical inputs (though sliders do this)
val_bed = validasi_numerik(in_bed, 'bedrooms')
val_bath = validasi_numerik(in_bath, 'bathrooms')
val_sqft = validasi_numerik(in_sqft, 'sqft_living')
val_flr = validasi_numerik(in_flr, 'floors')
val_abv = validasi_numerik(in_abv, 'sqft_above')

# Input fields for categorical features (using select boxes)
st.subheader('Input Fitur Kategorikal')

# City
city_classes = sorted(encoders['city'].classes_.tolist())
in_city = st.selectbox('Kota (City)', city_classes)
enc_city = encoders['city'].transform([in_city])[0]

# Statezip
statezip_classes = sorted(encoders['statezip'].classes_.tolist())
in_zip = st.selectbox('Kode Pos (Statezip)', statezip_classes)
enc_zip = encoders['statezip'].transform([in_zip])[0]

# Prepare input DataFrame
X_input = pd.DataFrame({
    'bedrooms': [val_bed],
    'bathrooms': [val_bath],
    'sqft_living': [val_sqft],
    'floors': [val_flr],
    'sqft_above': [val_abv],
    'city': [enc_city],
    'statezip': [enc_zip]
})

# Scale the input features
X_scaled = scaler.transform(X_input)

# Make prediction when button is clicked
if st.button('Prediksi Harga Rumah'):
    try:
        pred_skala = model.predict(X_scaled)
        # Ensure pred_skala is 2D for inverse_transform if it's 1D
        if pred_skala.ndim == 1:
            pred_skala = pred_skala.reshape(-1, 1)

        harga_asli = target_scaler.inverse_transform(pred_skala)
        st.success(f'Hasil Prediksi Harga Rumah: ${harga_asli[0][0]:,.2f}')
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
