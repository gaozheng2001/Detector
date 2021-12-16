'''
包含手和面部的检测
'''
import cv2
import time
import mediapipe as mp


if __name__ == "__main__":

    pTime = 0
    cTime = 0
    
    #获取窗口
    cap = cv2.VideoCapture(0)
    
    #手的detector
    mpHands = mp.solutions.hands
    hdetector = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    #面部的detector
    fdetector = cv2.CascadeClassifier('data\haarcascades\haarcascade_frontalface_default.xml')
    FACTOR = 1#缩放比例

    while(True):
        ret, frame = cap.read()

        res = cv2.resize(frame, None, fx=FACTOR, fy=FACTOR)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #检测手并绘制关键点
        results = hdetector.process(RGB)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    #print(id,lm)
                    h, w, c = frame.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    #if id ==0:
                    cv2.circle(frame, (cx,cy), 3, (255,0,255), cv2.FILLED)

                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
        
        #检测面部并绘制boundingbox
        faces_rects = fdetector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5)
        
        for (x, y, w, h) in faces_rects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        

        #计算帧率
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()