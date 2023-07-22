import cv2 as cv 

cam = cv.VideoCapture(0)
if cam.isOpened():
    print("Camera is open.")
else:
    print("Camera is not open.")


face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
while True:

    _, frame = cam.read()
    try: 
        objects = face_cascade.detectMultiScale(frame, 1.3, 5)
        x, y, w, h = objects[0]
        frame = cv.rectangle(frame, (x, y), (x+w, y+h), color = (255, 0, 0), thickness = 3)
        
    except Exception as E: 
        pass 

    cv.imshow('frame', frame)
    cv.waitKey(1)

    if 0xFF == ord('q'):
        break 

cam.release()
cv.destroyAllWindows()