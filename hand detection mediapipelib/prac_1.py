import cv2
import mediapipe as mp
import time

cap= cv2.VideoCapture(0)
mHand =mp.solutions.hands

hands = mHand.Hands()
mpdraw = mp.solutions.drawing_utils
while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    print(result.multi_hand_landmarks)

    if result.multi_hand_landmarks:
        for handmls in result.multi_hand_landmarks:
            mpdraw.draw_landmarks(img, handmls, mHand.HAND_CONNECTIONS)


    cv2.imshow("Image",img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
            break
cap.release()

cv2.destroyAllWindows()