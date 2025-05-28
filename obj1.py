import face_recognition
import streamlit as st
import pandas as pd
import cv2
import numpy as np
import csv
import os

st.title("Attendance System")
st.header('Trained with Face Recognition Algorithm')
st.header('Build with Streamlit ')

from datetime import datetime
 
video_capture = cv2.VideoCapture(0)
 
Kowshik_image = face_recognition.load_image_file("Kowshik.jpeg")
Kowshik_encoding = face_recognition.face_encodings(Kowshik_image)[0]

Vamsi_image = face_recognition.load_image_file("Vamsi.jpeg")
Vamsi_encoding = face_recognition.face_encodings(Vamsi_image)[0]

Nethra_image = face_recognition.load_image_file("Nethra.jpeg")
Nethra_encoding = face_recognition.face_encodings(Nethra_image)[0]

Harsha_image = face_recognition.load_image_file("Harsha.jpeg")
Harsha_encoding = face_recognition.face_encodings(Harsha_image)[0]




 
known_face_encoding = [

Kowshik_encoding,
Vamsi_encoding,
Nethra_encoding,
Harsha_encoding

]
 
known_faces_names = [
    
"Kowshik-4491",
"Vamsi-4471",
"Nethra-4474",
"Harsha-41645"
]
 
students = known_faces_names.copy()
 
face_locations = []
face_encodings = []
face_names = []
s=True
 
data = [["Name","Date"]]
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
 
file = current_date+'.csv'
f = open(current_date+'.csv','w+',newline = '')
lnwriter = csv.writer(f)
lnwriter.writerow(['Name','Time','Attendance'])
 
while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_faces_names[best_match_index]
 
            face_names.append(name)
            if name in known_faces_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,100)
                fontScale              = 1.5
                fontColor              = (255,0,0)
                thickness              = 3
                lineType               = 2
 
                cv2.putText(frame,name+' Present', 
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
 
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%HH-%MM-%SS")
                    
                    lnwriter.writerow([name,current_time,"Present"])
                
                    
                    
                    
                    
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for i in students:
    lnwriter.writerow([i,0,"Absent"]) 

video_capture.release()
cv2.destroyAllWindows()
f.close()
df = pd.read_csv(file)
st.table(df)