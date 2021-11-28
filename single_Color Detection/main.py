import imutils, time
import cv2

#hsv color code for upper and lower blue color
bluelower = (0, 193, 98)
blueupper= (179, 255, 255) 
# bluelower = (0,0,0)
# blueupper = (0,255,255)

# Start camera 0 for default camera
cam = cv2.VideoCapture(0)
time.sleep(1)
while True:
    (grabbed, frame) = cam.read()
    frame = imutils.resize(frame, width=1000)
    blurr = cv2.GaussianBlur(frame,(11,11),0)
    hsv = cv2.cvtColor(blurr, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, bluelower, blueupper)
    mask = cv2.erode(mask,None,iterations=2)
    mask = cv2.dilate(mask,None,iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    if len(cnts)>0:
        c = max(cnts, key=cv2.contourArea)
        ((x,y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"]/ M["m00"]), int(M["m01"]/M["m00"]))
        if radius >1:
            cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
            cv2.circle(frame,center,5,(0,0,255),-1)

            print(radius, center)
            if radius> 250:
                print("STOP")
            else:
                if(center[0]<150):
                    print("LEFT")
                elif(center[0]>450):
                    print("RIGHT")
                elif radius<250 :
                    print("FRONT")
                else:
                    print("Stop")

    cv2.imshow("FRAME",frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
            break
cam.release()

cv2.destroyAllWindows()
