import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from streamlit_drawable_canvas import st_canvas

def predictDigit(image):
    model = tf.keras.models.load_model("model/handwritten.h5")
    image = ImageOps.grayscale(image)
    img = image.resize((28,28))
    img = np.array(img, dtype='float32')
    img = img/255
    img = img.reshape((1,28,28,1))
    pred = model.predict(img)
    result = np.argmax(pred[0])
    return result

st.set_page_config(
    page_title='Reconocimiento de Dígitos',
    page_icon='✏️',
    layout='wide'
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Kanit&display=swap');

html, body, .stApp {
    background: linear-gradient(to right, #2c5364, #203a43, #0f2027);
    color: #ffffff;
    font-family: 'Kanit', sans-serif;
    text-align: center;
}

h1, h2, h3, h4, h5, h6, .stTitle, .stHeader {
    color: #ffcc70;
    text-align: center;
}

.stButton>button {
    background-color: #ffcc70;
    color: #1e1e1e;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    border: none;
}

.stSidebar {
    background-color: #ffffff22 !important;
}

.stImage>img {
    margin-left: auto;
    margin-right: auto;
}

.block-container {
    padding-left: 5%;
    padding-right: 5%;
}
</style>
""", unsafe_allow_html=True)

st.title('✏️ Reconocimiento de Dígitos Escritos a Mano')
st.subheader("🧠 Dibuja un dígito en el panel y presiona 'Predecir'")

drawing_mode = "freedraw"
stroke_width = st.slider('✍️ Selecciona el grosor del trazo', 1, 30, 15)
stroke_color = '#FFFFFF'
bg_color = '#000000'

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0.1)",
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=200,
    width=200,
    key="canvas",
)

if st.button('🔍 Predecir'):
    if canvas_result.image_data is not None:
        input_numpy_array = np.array(canvas_result.image_data)
        input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
        input_image.save('prediction/img.png')
        img = Image.open("prediction/img.png")
        res = predictDigit(img)
        st.header('🔢 El dígito es: ' + str(res))
    else:
        st.warning('⚠️ Por favor dibuja un número en el lienzo.')

st.sidebar.title("ℹ️ Acerca de")
st.sidebar.markdown("Esta app demuestra el reconocimiento de dígitos escritos a mano usando una red neuronal.")
st.sidebar.markdown("Modelo entrenado con **MNIST**.")
st.sidebar.markdown("Desarrollo basado en código de **Vinay Uniyal**.")
