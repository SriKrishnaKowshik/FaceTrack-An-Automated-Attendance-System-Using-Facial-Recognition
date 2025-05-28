import face_recognition
import pandas as pd
import cv2
import numpy as np
import csv
import os

from datetime import datetime
 
video_capture = cv2.VideoCapture(0)
 
BHarsha_image = face_recognition.load_image_file("BHarsha.jpeg")
BHarsha_encoding = face_recognition.face_encodings(BHarsha_image)[0]

Gopi_image = face_recognition.load_image_file("Gopi.jpeg")
Gopi_encoding = face_recognition.face_encodings(Gopi_image)[0]

Siva_image = face_recognition.load_image_file("Siva.jpeg")
Siva_encoding = face_recognition.face_encodings(Siva_image)[0]

Praveen_image = face_recognition.load_image_file("Praveen.jpeg")
Praveen_encoding = face_recognition.face_encodings(Praveen_image)[0]


 
known_face_encoding = [
    
BHarsha_encoding,
Gopi_encoding,
Siva_encoding,
Praveen_encoding

]
 
known_faces_names = [
    
"B.V.SS.Harsha-99210042195",
"C Gopi Sri vardhan-99210041845",
"S.Siva Kumar Reddy-99210042182 ",
"R. Praveen-99210042208"
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
                    
                    lnwriter.writerow([name,current_time,"P"])
                for i in students:
                    lnwriter.writerow([i,0,"A"])
                    
                    
                    
                    
    cv2.imshow("attendence system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

video_capture.release()
cv2.destroyAllWindows()
f.close()
