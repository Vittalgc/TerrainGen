import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from tensorflow.keras.saving import load_model
import cv2
import random

# st.set_page_config(layout = 'centered')

st.title("Heightmap Generation using DCCWGAN")
st.subheader("How to use: ")
st.write("1. Draw a sketch in the canvas provided below.")
st.write("2. Use a color picker to select different shades of grey/white/black. ")
st.write("3. The lighter the color that is used, the higher the elevation will be.")

# Specify canvas parameters in application
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("point", "freedraw", "line", "rect", "circle", "transform")
)

stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)
if drawing_mode == 'point':
    point_display_radius = st.sidebar.slider("Point display radius: ", 1, 25, 3)
stroke_color = st.sidebar.color_picker("Stroke color hex: ")
bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
bg_image = st.sidebar.file_uploader("Background image:", type=["png"])

realtime_update = st.sidebar.checkbox("Update in realtime", True)

# create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    background_image=cv2.imread(bg_image) if bg_image else None,
    update_streamlit=realtime_update,
    height=300,
    width=300,
    drawing_mode=drawing_mode,
    point_display_radius=point_display_radius if drawing_mode == 'point' else 0,
    key="canvas",
)

model = load_model("./Model/savedModel/model_0070.tf")

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype(np.float32), (128, 128))
    img_rescaling = (cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_NEAREST))
    x_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    arr = cv2.normalize(x_img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    height, width = arr.shape

    grid_width, grid_height = width // 4, height // 4

    imr_values = [0, 0.25, 0.5, 0.75, 1]

    IMR = np.zeros((grid_height, grid_width))
    
    for i in range(grid_height):
        for j in range(grid_width):
            start_i = i * 4
            end_i = min((i + 1) * 4, height)
            start_j = j * 4
            end_j = min((j + 1) * 4, width)

            area = arr[start_i:end_i, start_j:end_j]

            if area.size > 0:
                avg_brightness = np.mean(area)

                closest_values = sorted(imr_values, key=lambda x: abs(avg_brightness - x))[:2]
                assigned_value = random.choice(closest_values)
                IMR[i][j] = assigned_value
                
    IMR_reshaped = IMR.reshape((32, 32, 1))
    # imrs = [IMR_reshaped for i in range(5)]
    imrs = []
    imrs.append(IMR_reshaped)
    imr_in = np.array(imrs)

    z_vector = np.random.normal(-1, 1, (1, 128, 8 ,1))
    
    if st.button("Generate"):
        y_pred = model.predict([imr_in, z_vector])
        st.subheader("Generated Image")
        # col1, col2, col3, col4, col5 = st.columns(5)

        # col1.image(y_pred[0], clamp=True, output_format="PNG")
        # col2.image(y_pred[1], clamp=True, output_format="PNG")
        # col3.image(y_pred[2], clamp=True, output_format="PNG")
        # col4.image(y_pred[3], clamp=True, output_format="PNG")
        # col5.image(y_pred[4], clamp=True, output_format="PNG")

        st.image(y_pred, clamp=True, output_format="PNG")

















