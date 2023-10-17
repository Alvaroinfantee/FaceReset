import streamlit as st
import face_recognition
from PIL import Image
import numpy as np

# Initialize database
database = {
    "Joe": {
        "balance": 1000,
        "face_encoding": face_recognition.face_encodings(face_recognition.load_image_file('rogan1.jpeg'))[0]
    },
    "Dillon": {
        "balance": 1000,
        "face_encoding": face_recognition.face_encodings(face_recognition.load_image_file(r"C:\Users\alvar\Downloads\Dillon2.jpg"))[0]
    }
}

def deduct_balance_from_user(uploaded_image):
    # Convert the file to a numpy array
    uploaded_image = np.array(uploaded_image)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    uploaded_image = uploaded_image[:, :, ::-1]
    
    uploaded_face_encodings = face_recognition.face_encodings(uploaded_image)
    if len(uploaded_face_encodings) == 0:
        st.write("No faces found in the image.")
        return
    for uploaded_face_encoding in uploaded_face_encodings:
        for person, data in database.items():
            match = face_recognition.compare_faces([data["face_encoding"]], uploaded_face_encoding)
            if match[0]:
                database[person]["balance"] -= 100
                st.write(f"Recognized {person}. Deducted 100 from their balance. New balance: {database[person]['balance']}")
                return
    st.write("No matching faces found in the database.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)
if uploaded_file is not None:
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.write("Classifying...")
    deduct_balance_from_user(uploaded_image)
