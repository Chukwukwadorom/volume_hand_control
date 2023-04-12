import cv2
import numpy as np
import time
from HandTrackModule import HandDetection


from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volume.GetVolumeRange()
#volume.SetMasterVolumeLevel(-20.0, None)

cap = cv2.VideoCapture(0)
# cap.set(3, 1920)
# cap.set(4, 1080)

MIN_VOL, MAX_VOL = volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]


prev_time = 0

hand_detection= HandDetection(min_detection_con = 0.9)

MIN_LINE_LEN = 40
MAX_LINE_LEN = 300
VOLBAR = 400
VOLPER = 0

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cur_time = time.time()
    fps  = int(1/(cur_time - prev_time))
    prev_time = cur_time

    if not success:
        print("did not read frame")
        break

    cv2.putText(frame, f'FPS: {fps}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)
    frame = hand_detection.get_hands(frame, draw=False)
    position = hand_detection.get_positions(frame, draw=False)
    if len(position) > 0:
        thumb, index = position[4], position[8]
        x_t, y_t = thumb[1], thumb[2]
        x_i, y_i = index[1], index[2]

        cv2.circle(frame, (x_t, y_t), 10, (0, 0, 255), -1)
        cv2.circle(frame, (x_i, y_i), 10, (0, 0, 255), -1)

        ### draw a line between thumb and index ###
        cv2.line(frame, (x_t, y_t), (x_i, y_i), (55,175,212), 2)

        ##length of line ##
        LIN_lEN = np.hypot(x_t - x_i, y_t - y_i)

        ##midpoint of line ##
        x_m, y_m = int((x_t + x_i)//2), int((y_t + y_i)//2)
        cv2.circle(frame, (x_m, y_m), 5, (0, 0, 255), -1)

        if LIN_lEN <= MIN_LINE_LEN:
            cv2.circle(frame, (x_m, y_m), 10, (0, 255, 0 ), -1)


        ## infer/set volume  from line
        vol = np.interp(LIN_lEN, [MIN_LINE_LEN,MAX_LINE_LEN], [MIN_VOL, MAX_VOL])
        volume.SetMasterVolumeLevel(vol, None)

        ##cal vol bar/percentage
        VOLBAR = np.interp(LIN_lEN, [MIN_LINE_LEN, MAX_LINE_LEN], [400, 200])
        VOLPER = int(np.ceil(np.interp(LIN_lEN, [MIN_LINE_LEN, MAX_LINE_LEN], [0, 100])))



    ## draw volume bar ##
    frame = cv2.rectangle(frame, (20, 200), (45, 400), color=(255, 0, 0), thickness=2)
    frame = cv2.rectangle(frame, (20, int(VOLBAR)), (45, 400), color=(255, 0, 0), thickness=-1)
    cv2.putText(frame, f'{VOLPER} %', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)


    cv2.imshow("image", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()