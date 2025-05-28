import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known face images and encodings
def load_face(image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0]
    else:
        raise ValueError(f"No face found in the image '{image_path}'.")

try:
    known_faces = {
        "Kowshik-4491": load_face("W:\AIML\Attendence\kowshik.jpeg"),
        "Vamsi-4471": load_face("Vamsi.jpeg"),
        "Harsha-41645": load_face("Harsha.jpeg"),
        "Shyam-4223": load_face("Shyam.jpeg")
    }
except ValueError as e:
    print(e)
    exit()

students = list(known_faces.keys())

# Initialize CSV file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

with open(f'{current_date}.csv', 'w+', newline='') as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Date"])

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
            face_distances = face_recognition.face_distance(list(known_faces.values()), face_encoding)
            best_match_index = np.argmin(face_distances)
            
            if matches[best_match_index]:
                name = list(known_faces.keys())[best_match_index]
                if name in students:
                    students.remove(name)
                    current_time = datetime.now().strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
                    
                    # Display name on video
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(frame, f'{name} Present', (10, 100), font, 1.5, (255, 0, 0), 3, cv2.LINE_AA)
        
        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
