#Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


#Loading the Model
model = load_model('./plant_disease.h5')

#Name of Classes
CLASS_NAMES = ['Corn_(maize)___Common_rust_', 'Potato___Early_blight', 'Tomato___Bacterial_spot']

#Setting Title of App
st.title("Plant Disease Detection")
st.markdown("Upload an image of the plant leaf")

#Uploading the dog image
plant_image = st.file_uploader("Choose an image...", type=["jpg", "png", "gif"])

#On file upload
if plant_image is not None:

    # Error handling
    try:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))

        #Convert image to 4 Dimension
        opencv_image = np.expand_dims(opencv_image, axis=0)

        #Make Prediction
        with st.spinner('Making prediction...'):
            Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]

        # Display the image and prediction
        st.image(opencv_image[0], channels="BGR")
        #st.success(f"Predicted class: {result}, confidence: {np.max(Y_pred):.2f}")
        st.write("Predicted classes and probabilities:")
        for i in range(len(CLASS_NAMES)):
            st.write(f"{CLASS_NAMES[i]}: {Y_pred[0][i]:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")