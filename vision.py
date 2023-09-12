import numpy as np
import cv2
import time
import tensorflow as tf
# from matplotlib import pyplot as plt


face_detection = cv2.CascadeClassifier('haar_cascade_face_detection.xml')

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
settings = {
	'scaleFactor': 1.3, 
	'minNeighbors': 5, 
	'minSize': (50, 50)
}

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("{0}:{1}:{2}".format(int(hours),int(mins),sec))


labels = str('face')




while True:
	ret, img = camera.read()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	detected = face_detection.detectMultiScale(gray, **settings)
    
	for x, y, w, h in detected:
		cv2.rectangle(img, (x, y), (x+w, y+h), (245, 135, 66), 2)
		cv2.rectangle(img, (x, y), (x+w//3, y+20), (245, 135, 66), -1)
		face = gray[y+5:y+h-5, x+20:x+w-20]
		face = cv2.resize(face, (48,48)) 
		face = face/255.0
		
  
		
		state = labels
		font = cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(img,state,(x+20,y+20), font, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        
        
    
	cv2.imshow('Facial Expression', img)

	if cv2.waitKey(5) != -1:
		break

camera.release()
cv2.destroyAllWindows()
