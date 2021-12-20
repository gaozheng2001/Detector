import math
import cv2
import mediapipe as mp
import time

def finger_check(landmark):

    if len(landmark) < 21:
        return 'bad detection'
    
    #wriet = landmark[0]
    thumb = landmark[1:5]
    index = landmark[5:9]
    middle = landmark[9:13]
    ring = landmark[13:17]
    pinky = landmark[17:]

    str_fingers = [straight_finger(thumb), straight_finger(index), straight_finger(middle), straight_finger(ring), straight_finger(pinky)]

    if is_straight_finger(str_fingers, [1]):
        if finger_up(index):
            return 'up'

    return ''

def straight_finger(finger) :
    error_angle = 10

    mcp, pip, dip, tip = finger[0], finger[1], finger[2], finger[3]

    k1x, k1y = (mcp.x - pip.x)/distance(mcp, pip), (mcp.y - pip.y)/distance(mcp, pip)
    k2x, k2y = (pip.x - dip.x)/distance(pip, dip), (pip.y - dip.y)/distance(pip, dip)
    k3x, k3y = (dip.x - tip.x)/distance(dip, tip), (dip.y - tip.y)/distance(dip, tip)
    
    if abs(k1x*k2y - k2x*k1y) < math.sin(math.pi*error_angle/180) and abs(k3x*k2y - k2x*k3y) < math.sin(math.pi*error_angle/180):
        return True
    else:
        return False

def is_straight_finger(fingers, require):
    result = False
    for index in range(len(require)):
        if fingers[require[index]]:
            result = True
    return result


def finger_up(finger):
    error_angle = 10

    mcp,tip = finger[0], finger[3]
    ky = (tip.y - mcp.y)/distance(mcp, tip)
    if -ky > math.sin(math.pi*(1/2 - error_angle/180)):
        return True
    return False

def distance(point1, point2):
    return math.sqrt(math.pow(point1.x - point2.x, 2) + math.pow(point1.y - point2.y, 2))

if __name__ == '__main__':
    
    cap = cv2.VideoCapture(0)
    
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(static_image_mode=False,
                          max_num_hands=2,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)
    mpDraw = mp.solutions.drawing_utils
    
    pTime = 0
    cTime = 0

    detected = 0
    
    while True:
        detected += 1

        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                str_fingers = finger_check(handLms.landmark)
                print('new check')
                print(str_fingers)
                if detected%15 == 0:
                    print('30fpes')
                    

                for id, lm in enumerate(handLms.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    
                    cv2.circle(img, (cx,cy), 3, (255, 0, 255), cv2.FILLED)
    
                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
    
        cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    
        cv2.imshow("Image", img)
    
        if cv2.waitKey(100) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

