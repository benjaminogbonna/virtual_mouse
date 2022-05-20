import mediapipe as mp
import cv2
import time
import autopy
import numpy as np
import hand_tracking_module as htm

cam_w, cam_h = 640, 480
screen_width, screen_height = autopy.screen.size()
frame_reduction = 100
smoothening = 5
prev_time = 0
prev_loc_x, prev_loc_y = 0, 0
cur_loc_x, cur_loc_y = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, cam_w)
cap.set(4, cam_h)
detector = htm.HandDetector(max_hands=1, detection_con=.8)


while True:
    # find hand land marks
    success, img = cap.read()
    img = detector.find_hands(img)
    # get the tip of the index and middle fingers
    land_mark, bound_box = detector.find_position(img, draw=True)

    if len(land_mark) > 0:
        x1, y1 = land_mark[8][1:]
        x2, y2 = land_mark[12][1:]

        # check which fingers are up
        fingers = detector.fingers_up()
        cv2.rectangle(img, (frame_reduction, frame_reduction), (cam_w - frame_reduction, cam_h - frame_reduction),
                      (255, 0, 255), 2)
        # only index finger: moving mode
        if fingers[1] == 1 and fingers[2] == 0:

            # convert coordinates
            x3 = np.interp(x1, (frame_reduction, cam_w-frame_reduction), (0, screen_width))
            y3 = np.interp(y1, (frame_reduction, cam_h-frame_reduction), (0, screen_height))

            # smoothen the values
            cur_loc_x = prev_loc_x + (x3 - prev_loc_x) /smoothening
            cur_loc_y = prev_loc_y + (y3 - prev_loc_y) / smoothening

            autopy.mouse.move(screen_width - cur_loc_x, cur_loc_y)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            prev_loc_x, prev_loc_y = cur_loc_x, cur_loc_y
        # check if both index and middle fingers are up
        if fingers[1] == 1 and fingers[2] == 1:
            # find distance between fingers
            length, img, line_info = detector.find_distance(8, 12, img)
            if length < 40:
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                # click mouse if distance is short
                autopy.mouse.click()
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time

    cv2.putText(img, str(int(fps)), (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
