import face_recognition
import cv2 
import numpy
import os
from datetime import datetime


frame_count = 0
path = 'Attendances'
images = []
myList = os.listdir(path)
class_name = []


for cls in myList : 
    curimage = cv2.imread(f'{path}/{cls}')
    images.append(curimage)
    class_name.append(os.path.splitext(cls)[0])



def finding_encoding (images):
    encode_list = []
    for img in images :
        cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

def mark(name):
    with open('Attendance.csv', 'r') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.strip().split(',')  # حذف \n
            name_list.append(entry[0])
    
    if name not in name_list:
        now = datetime.now()
        dtstring = now.strftime('%H:%M:%S')
        with open('Attendance.csv', 'a') as f:
            f.write(f'\n{name},{dtstring}')


mark('Elon')  

knownlist = finding_encoding(images)
print('Encoding Completed')

cap = cv2.VideoCapture(0)

while True : 
    success , img =  cap.read()



    imgS = cv2.resize(img , (0,0) , None , 0.25 , 0.25)
    imgS = cv2.cvtColor(imgS , cv2.COLOR_BGR2RGB)

    curEncode = face_recognition.face_encodings(imgS)
    curLoc    = face_recognition.face_locations(imgS)


    for encodeFace , locFace in zip(curEncode , curLoc) :

        matches = face_recognition.compare_faces(knownlist , encodeFace) 
        faceDis = face_recognition.face_distance(knownlist , encodeFace)
        index_match = numpy.argmin(faceDis) 

        if matches[index_match] :
            name = class_name[index_match]

            y1 , x2 , y2 , x1 = locFace 
            y1 , x2 , y2 , x1 = y1*4 , x2*4 , y2*4 , x1*4

            cv2.rectangle(img ,(x1,y1) , (x2,y2) , (223 , 247 , 76)  , 2)
            cv2.rectangle(img ,(x1,y2 - 35) , (x2,y2) , (223 , 247 , 76) , cv2.FILLED)
            cv2.putText(img , name , (x1+6,y2-6) ,cv2.FONT_HERSHEY_PLAIN , 2 , (81, 227 , 59),3)

            mark(name)

    
    key = cv2.waitKey(1)
    
    if key == ord('q') :
        break

    cv2.imshow('Webcam' , img)
    

cap.release()
cv2.destroyAllWindows()




