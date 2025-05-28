import face_recognition
import streamlit as st
import pandas as pd
import cv2
import numpy as np
import csv
import os
from datetime import datetime

st.title("Attendance System")
st.header('Trained with Face Recognition Algorithm')
st.header('Built with Streamlit')

def load_face_encoding(image_path, name):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) > 0:
        return encodings[0]
    else:
        st.error(f"No faces found in the image for {name}. Please check the image file.")
        return None

# Load images and encode faces
BHarsha_encoding = load_face_encoding("BHarsha.jpeg", "B.V.SS.Harsha")
Gopi_encoding = load_face_encoding("Gopi.jpeg", "C Gopi Sri vardhan")
Siva_encoding = load_face_encoding("Siva.jpeg", "S.Siva Kumar Reddy")
Praveen_encoding = load_face_encoding("Praveen.jpeg", "R. Praveen")

# Filter out any None encodings
known_face_encodings = [encoding for encoding in [
    BHarsha_encoding, Gopi_encoding, Siva_encoding, Praveen_encoding
] if encoding is not None]

known_face_names = [
    "B.V.SS.Harsha-99210042195",
    "C Gopi Sri vardhan-99210041845",
    "S.Siva Kumar Reddy-99210042182",
    "R. Praveen-99210042208"
]

students = known_face_names.copy()

# Initialize variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# Prepare CSV file for recording attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
file = current_date + '.csv'
with open(file, 'w+', newline='') as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(['Name', 'Time', 'Attendance'])

# Start video capture
video_capture = cv2.VideoCapture(0)

stframe = st.empty()

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
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
                # Draw label
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10, 100)
                fontScale = 1.5
                fontColor = (255, 0, 0)
                thickness = 3
                lineType = 2
                cv2.putText(frame, name + ' Present', bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

                if name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H:%M:%S")
                    with open(file, 'a', newline='') as f:
                        lnwriter = csv.writer(f)
                        lnwriter.writerow([name, current_time, "Present"])

    process_this_frame = not process_this_frame

    # Display the resulting frame
    stframe.image(frame, channels="BGR")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Mark remaining students as absent
with open(file, 'a', newline='') as f:
    lnwriter = csv.writer(f)
    for student in students:
        lnwriter.writerow([student, "N/A", "Absent"])

# Release the video capture
video_capture.release()
cv2.destroyAllWindows()

# Display the attendance table in Streamlit
df = pd.read_csv(file)
st.table(df)
