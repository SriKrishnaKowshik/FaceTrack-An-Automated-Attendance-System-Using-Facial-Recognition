import face_recognition
import streamlit as st
import pandas as pd
import cv2
import numpy as np
import csv
from datetime import datetime

st.title("Attendance System")
st.header('Trained with Face Recognition Algorithm')
st.header('Built with Streamlit')

def load_face_encoding(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            return encodings[0]
    except Exception as e:
        st.error(f"Error loading image {image_path}: {e}")
    return None

known_face_encodings = [
    load_face_encoding("Kowshik.jpeg"),
    load_face_encoding("Vamsi.jpeg"),
    load_face_encoding("Harsha.jpeg"),
    load_face_encoding("Shyam.jpeg")
]

known_face_names = [
    "Kowshik-4491",
    "Vamsi-4471",
    "Harsha-41645",
    "Shyam-4223"
]

students = known_face_names.copy()

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
file = current_date + '.csv'
with open(file, 'w+', newline='') as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(['Name', 'Time', 'Attendance'])

video_capture = cv2.VideoCapture(0)
stop_button_pressed = False

while not stop_button_pressed:
    ret, frame = video_capture.read()
    if not ret:
        st.warning("Failed to grab frame from video feed")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)
        if name in known_face_names:
            if name in students:
                students.remove(name)
                current_time = datetime.now().strftime("%H-%M-%S")
                with open(file, 'a', newline='') as f:
                    lnwriter = csv.writer(f)
                    lnwriter.writerow([name, current_time, "P"])

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(frame_rgb, channels="RGB", use_column_width=True)

    if st.button("Stop Video"):
        stop_button_pressed = True

for student in students:
    with open(file, 'a', newline='') as f:
        lnwriter = csv.writer(f)
        lnwriter.writerow([student, "N/A", "A"])

video_capture.release()
cv2.destroyAllWindows()

df = pd.read_csv(file)
st.table(df)
