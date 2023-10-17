import streamlit as st
import cv2
import numpy as np
import tempfile

def deduct_balance_from_user(uploaded_image_path, deduction_amount):
    uploaded_image = cv2.imread(uploaded_image_path, cv2.IMREAD_GRAYSCALE)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(uploaded_image, 1.3, 5)

    if len(faces) == 0:
        st.write("No faces found in the image.")
        return

    for (x, y, w, h) in faces:
        face_region = uploaded_image[y:y+h, x:x+w]
        label, confidence = recognizer.predict(face_region)

        if confidence < 100:
            person = list(database.keys())[label]
            old_balance = database[person]["balance"]
            database[person]["balance"] -= deduction_amount
            new_balance = database[person]["balance"]
            st.write(f"Recognized {person}. Deducted {deduction_amount} from their balance. Old balance: {old_balance}, New balance: {new_balance}")
            return

    st.write("No matching faces found in the database.")

# Initialize recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

database = {
    "Joe": {
        "balance": 1000,
        "face_encoding": cv2.imread(r"C:\Users\alvar\Downloads\rogan1.jpg", cv2.IMREAD_GRAYSCALE)
    },
    "Dillon": {
        "balance": 1000,
        "face_encoding": cv2.imread(r"C:\Users\alvar\Downloads\Dillon2.jpg", cv2.IMREAD_GRAYSCALE)
    }
}

# Train recognizer on initial data
labels = []
faces = []
for i, (name, data) in enumerate(database.items()):
    labels.append(i)
    faces.append(data['face_encoding'])

recognizer.train(faces, np.array(labels))

st.title("Face Recognition WebApp")
st.write("## Upload an image for face recognition")

uploaded_file = st.file_uploader("Choose a JPG file", type="jpg")

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    st.write("### Uploaded Image:")
    st.image(tfile.name)

    deduction_amount = st.number_input("Enter the amount to deduct", min_value=0, max_value=1000, value=100)
    
    if st.button("Process"):
        deduct_balance_from_user(tfile.name, deduction_amount)

