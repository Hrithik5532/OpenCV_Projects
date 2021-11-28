import  cv2, os

haar_file = "haarcascade_frontalface_default.xml"
dataset = "dataset"
sub_data = "sahil"

path = os.path.join(dataset,sub_data)
if not os.path.isdir(path):
    os.mkdir(path)

(width, height) = (130,100)

face_casscade = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(0)

count = 1
while count <31:
    print(count)
    (_, img)= cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_casscade.detectMultiScale(gray)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0),2)
        face= gray[y:y+h, x:x+h]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite(f'{path}/{count}.png',face_resize)
    count += 1

    cv2.imshow("Saving in DataBase", img)
    key = cv2.waitKey(100)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()