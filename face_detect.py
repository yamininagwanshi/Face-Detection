import cv2

#Load the cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#to capture video form webcam
cap=cv2.VideoCapture(0)

#to use video as input
#cap=cv2.VideoCapture("filename.mp4")

while True:
    #Read the frames
    _, image = cap.read() # underscore will used to read frame

    #convert to grayscale
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #Detect the faces
    faces = face_cascade.detectMultiScale(gray,1.1, 4)

    #Draw the rect around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y), (x+w,y+w), (189,0,0), 3)

    #display
    cv2.imshow("Face Detection", image)

    #Stoping Condition
    k = cv2.waitKey(10) & 0xff
    if (k==27):
        break

#release the videocapture object
 
cv2.destroyAllWindows()



