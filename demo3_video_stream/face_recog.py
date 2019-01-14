import cv2
import sys

cascPath = '/usr/local/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)
while True:
    _, original_frame = video_capture.read()

    frame = cv2.resize(original_frame, (0,0), fx=0.5, fy=0.5)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(33) == 0x1b:
        break

video_capture.release()
cv2.destroyAllWindows()
