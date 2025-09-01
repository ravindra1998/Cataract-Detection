import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ✅ Must be first Streamlit command
# st.set_page_config(page_title="Cataract Detection", layout="wide")
st.set_page_config(page_title="Cataract Detection", page_icon="Ravi.jpg",layout="wide"  )  # can be PNG, JPG, or emoji
# Display the same image inside the app
st.image("Ravi.jpg", width=200)
# st.title("Cataract Detection")
# Constants
IMG_SIZE = 224
EYE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_eye.xml"

# Load model
@st.cache_resource
def load_cataract_model():
    model = load_model("cataract_model.h5")
    return model

model = load_cataract_model()

# Prediction function
def predict(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0][0]
    label = "Normal" if prediction > 0.5 else "Cataract"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Detect eyes and classify each one
def detect_and_classify_eyes(frame, model, eye_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in eyes:
        roi_color = frame[y:y+h, x:x+w]
        eye_img = cv2.resize(roi_color, (IMG_SIZE, IMG_SIZE))
        eye_img = np.array(eye_img) / 255.0
        eye_img = np.expand_dims(eye_img, axis=0)

        pred = model.predict(eye_img)[0][0]
        label = "Cataract" if pred > 0.5 else "Normal"
        color = (0, 0, 255) if label == "Cataract" else (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

# ---------------------------- UI ----------------------------

st.title("👁️ Cataract Detection App")
st.caption("Developed by: Ravindra Singh")

# Sidebar options
option = st.sidebar.radio("Choose mode", ("Upload Image", "Live Camera"))

# Load eye classifier
eye_cascade = cv2.CascadeClassifier(EYE_CASCADE_PATH)

# ------------------------ Upload Image ------------------------
if option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        st.write("Predicting...")
        label, confidence = predict(image)

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence:.2%}")

# ------------------------ Live Camera ------------------------
elif option == "Live Camera":
    st.warning("Click 'Start' to launch webcam. Press 'q' in the video window to stop.")

    if st.button("Start Live Detection"):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not open webcam.")
        else:
            st.info("Press 'q' to stop the webcam window.")
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame.")
                    break

                frame = detect_and_classify_eyes(frame, model, eye_cascade)
                cv2.imshow("Live Cataract Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

