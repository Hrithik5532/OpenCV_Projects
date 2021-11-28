import cv2
import os
# dataset = "dataset"
# name = "champ"
# path = os.path.join(dataset,name)
# if not os.path.isdir(path):
#     os.makedirs(path)

width, height = 130,100
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

#start camera
cam = cv2.VideoCapture(0)

count = 1
while True:
    print(count)
    _,img =cam.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #this part will detect the face
    face = haar_cascade.detectMultiScale(grayImg)
    for (x,y,w,h) in face:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        faceOnly = grayImg[y:y+h,x:x+w]
        #resizeImg = cv2.resize(faceOnly,(width,height))
        count +=1
    cv2.imshow("Face Detection", img)
    key = cv2.waitKey(4)

    if key == 4:
        break
print("Image Captured succssesfully")
cam.release()
cv2.destroyAllWindows()