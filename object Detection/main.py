import numpy as np
import imutils
import cv2
import time

prototxt = "Feb12_MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
confTresh = 0.2

CLASSES = ["backgeound","aeroplane","bicycle","bird","boat",
            "bottle","bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor", "mobile"]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES),3))

print("Loading model........")
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("Model Loaded")
print("Starting Camera Feed .......")
vs = cv2.VideoCapture(0)
time.sleep(2.0)

while True:
    _, frame = vs.read()
    frame = imutils.resize(frame, width=500)
    img = frame.copy()
    (h, w)= img.shape[:2]
    imResize = cv2.resize(frame, (300, 500))
    blob = cv2.dnn.blobFromImage(imResize, 0.007843, (300,300), 127.5)
    net.setInput(blob)

    detections =net.forward()
    detshap= detections.shape[2]
    for i in np.arange(0, detshap):
        confidence = detections[0,0,i, 2]
        if confidence > confTresh:
            idx= int(detections[0,0,i,1])
            box = detections[0,0,i, 3:7]* np.array([w,h,w,h])
            (startX, StartY, endX, endY) =box.astype("int")

            label = "{}: {:.2f}%".format(CLASSES[idx],confidence*100)
            cv2.rectangle(frame, (startX, StartY),(endX, endY),COLORS[idx], 2)

            if StartY-15 >15:
                y =StartY - 15
            else :
                y= StartY + 15
            cv2.putText(frame, label, (startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.5, COLORS[idx],2)
    
    cv2.imshow("frame", frame)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
cv2.release()
cv2.destroyAllWindows()
