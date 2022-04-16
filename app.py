import streamlit as st
import io
import numpy as np
from PIL import Image
import tensorflow as tf
import efficientnet.tfkeras

import os
#### For debug
# print("\n\n\n\n")
# print(os.listdir(os.getcwd()))
# print("\n\n\n\n")

# Title and Description
st.title('Plant Disease Detection')
st.write('Just upload your Plant\'s Leaf!')

model = tf.keras.models.load_model('model.h5')

# Upload the image
uploaded_file = st.file_uploader('Choose your image', type=['png', 'jpg'])

predictions_map = {0:'is Healthy', 1:'Has disease (Scab)'}

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, use_column_width=True)
    # Image preprocessing
    resized_image = np.array(image.resize((512,512)))/255. # Resize image and divide pixel number by 255. for having values between 0 and 1 (normalize it)
    # Adding batch dimensions
    image_batch = resized_image[np.newaxis, :, :, :]
    # Getting the predictions from the model
    predictions_arr = model.predict(image_batch)
    predictions = np.argmax(predictions_arr)
    
    if predictions>=1:
        predictions =1
    result_text = f'The plant leaf {predictions_map[predictions]}'
    
    if predictions == 0:
        st.success(result_text)
    else:
        st.error(result_text)
