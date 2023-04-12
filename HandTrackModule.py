import cv2
import mediapipe as mp
import time



mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils



class HandDetection:
    def __init__(self, static_image_mode = False, max_num_hands = 2, model_complexity = 1,
                 min_detection_con = 0.5, min_tracking_con = 0.5):

        self.STATIC_IMAGE_MODE = static_image_mode
        self.MAX_NUM_HANDS = max_num_hands
        self.MODEL_COMPLEXITY = model_complexity
        self.MIN_DETECTION_CONFIDENCE = min_detection_con
        self.MIN_TRACKING_CONFIDENCE = min_tracking_con

        self.hands = mpHands.Hands(self.STATIC_IMAGE_MODE, self.MAX_NUM_HANDS, self.MODEL_COMPLEXITY,
                                   self.MIN_DETECTION_CONFIDENCE, self.MIN_TRACKING_CONFIDENCE)


    def get_hands(self, img, draw = True):
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.result = self.hands.process(frame_rgb)
        if self.result.multi_hand_landmarks:
            for hand in self.result.multi_hand_landmarks:
                if draw:
                    mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

        return img

    def get_positions(self, img, hand_no = 0, draw=True):
        positions = []
        if self.result.multi_hand_landmarks:
            hand = self.result.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(hand.landmark):
                x, y = lm.x, lm.y
                h, w, c = img.shape
                cx, cy = int(w*x), int(h*y)
                positions.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 3, (0,0, 255), -1)

        return positions






def main():
    prev_time = 0

    cap = cv2.VideoCapture(0)
    hand_detection = HandDetection()

    while True:
        success, frame = cap.read()
        cur_time = time.time()
        fps = int(1 / (cur_time - prev_time))
        prev_time = cur_time

        if not success:
            break
        hands_img = hand_detection.get_hands(frame)
        positions = hand_detection.get_positions(hands_img)
        if len(positions) > 0:
            print(positions[4])
        cv2.putText(frame, str(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("image", hands_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()