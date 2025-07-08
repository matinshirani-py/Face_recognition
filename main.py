import face_recognition
import cv2 
import numpy

imgElon = face_recognition.load_image_file('images/Elon_Musk.jpg')
imgElon = cv2.cvtColor(imgElon , cv2.COLOR_BGR2RGB )

imgElon_test = face_recognition.load_image_file('images/Elon_Musk_test.jpg')
imgElon_test = cv2.cvtColor(imgElon_test , cv2.COLOR_BGR2RGB )

faceLoc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon , (faceLoc[3] , faceLoc[0]), (faceLoc[1],faceLoc[2]),(0,0,255),2)

faceLoc_test = face_recognition.face_locations(imgElon_test)[0]
encodeElon_test = face_recognition.face_encodings(imgElon_test)[0]
cv2.rectangle(imgElon_test , (faceLoc_test[3] , faceLoc_test[0]), (faceLoc_test[1],faceLoc_test[2]),(0,0,255),1)


results = face_recognition.compare_faces([encodeElon] , encodeElon_test)
faceDis = face_recognition.face_distance([encodeElon] , encodeElon_test)

print(results , faceDis)


#cv2.imshow('Elon' , imgElon_test)
# cv2.imshow('Elontest' , imgElon_test)

cv2.waitKey(0)