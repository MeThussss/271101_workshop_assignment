import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# การรับ Input จากกล้อง webcam
cap = cv2.VideoCapture(1)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
    
      continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    #ให้ระบบ Detect นิ้วมือจากกล้อง 
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    #เซ็ตตัวแปรจำนวนนับของนิ้วมือให้เริ่มต้นที่ 0
    fingerCount = 0

    if results.multi_hand_landmarks:

      for hand_landmarks in results.multi_hand_landmarks:
        #ตรวจเช็คมือว่าเป็นข้างซ้ายหรือข้างหวา
        handIndex = results.multi_hand_landmarks.index(hand_landmarks)
        handLabel = results.multi_handedness[handIndex].classification[0].label

        #สร้างตัวแปรเพื่อเก็บค่าตำแหน่ง x และ y
        handLandmarks = []

        #ทำการเก็บค่าตำแหน่ง x และ y ลงในตัวแปร handLandmarks 
        for landmarks in hand_landmarks.landmark:
          handLandmarks.append([landmarks.x, landmarks.y])

        
        # : นับนิ้วโป้ง
        if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
          fingerCount = fingerCount+1
        elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
          fingerCount = fingerCount+1

        if handLandmarks[8][1] < handLandmarks[6][1]:       #นับนิ้วชี้
          fingerCount = fingerCount+1
        if handLandmarks[12][1] < handLandmarks[10][1]:     #นับนิ้วกลาง
          fingerCount = fingerCount+1
        if handLandmarks[16][1] < handLandmarks[14][1]:     #นับนิ้วนาง
          fingerCount = fingerCount+1
        if handLandmarks[20][1] < handLandmarks[18][1]:     #นับนิ้วก้อย 
          fingerCount = fingerCount+1

        # เปลี่ยนสีของเส้นสำหรับตรวจจับรูปมือ 
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    #ตัวเลขแสดงผลจำนวนนิ้วมือ 
    cv2.putText(image, str(fingerCount), (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 20, 147), 10)

    cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()