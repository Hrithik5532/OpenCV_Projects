import cv2, numpy as np , os
from cv2 import data
from numpy.lib.function_base import gradient

haar_file = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(haar_file)

dataset ="dataset"
print("Training.........!!!!!!!!!..........")
(images, labels, name, id )=([], [], {}, 0)

for (subdirs, dirs, files) in os.walk(dataset):
    for subdir in dirs:
        name[id] = subdir
        subjectpath = os.path.join(dataset, subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(label)
        id +=1

(images, labels) = [np.array(lis) for lis in [images, labels]]
print(images, labels)
(width, height)= (130, 100)

model =cv2.face.LBPHFaceRecognizer_create()

model.train(images, labels)
cam = cv2.VideoCapture(0)
cnt=0

while True:
    (_, img)= cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(255,255,0),2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width,height))

        prediction = model.predict(face_resize)
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),3)
        if prediction[1]<100:
            cv2.putText(img,f"{name[prediction[0]]}-{prediction[1]}",(x-10, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            print(name[prediction[0]])
            cnt=0
        else:
            cnt +=1
            cv2.putText(img, "Unknown",(x-10,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            print("Unknown Person")
            cv2.imwrite("unknown.jpg",img)
            cnt = 0
    cv2.imshow("FaceRecognition",img)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
cam.release()
cv2.destroyAllWindows()
