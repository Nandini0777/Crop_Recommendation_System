import streamlit as st
import pandas as pd
import pickle
from PIL import Image

# Title
st.title("🌱 Crop Recommendation System")
st.write("Enter the required soil and climate values below:")

# Load trained model
model = pickle.load(open('crop_model.pkl', 'rb'))

# Load crop descriptions
crop_data = pd.read_csv('crop_details.csv')
crop_data['crop'] = crop_data['crop'].str.strip().str.lower()

# Inputs from user
N = st.number_input("Nitrogen (N)", min_value=0)
P = st.number_input("Phosphorus (P)", min_value=0)
K = st.number_input("Potassium (K)", min_value=0)
temperature = st.number_input("Temperature (°C)", format="%.2f")
humidity = st.number_input("Humidity (%)", format="%.2f")
ph = st.number_input("pH", format="%.2f")
rainfall = st.number_input("Rainfall (mm)", format="%.2f")

# On button click
if st.button("Recommend Crop"):
    # Predict
    input_data = [[N, P, K, temperature, humidity, ph, rainfall]]
    prediction = model.predict(input_data)
    crop_name = prediction[0].strip().lower()

    # ✅ Show crop name
    st.subheader(f"🌾 Recommended Crop: **{crop_name.capitalize()}**")

    # ✅ Show crop image
    image_path = f"crop_images/{crop_name}.jpg"
    try:
        image = Image.open(image_path)
        st.image(image, caption=crop_name.capitalize(), use_container_width=True)
    except FileNotFoundError:
        st.error(f"❌ Image not found: `{image_path}`")
    except Exception as e:
        st.error(f"❌ Error loading image: {str(e)}")


    # ✅ Show crop description
    match = crop_data[crop_data['crop'] == crop_name]
    if not match.empty:
        st.info(match['description'].values[0])
    