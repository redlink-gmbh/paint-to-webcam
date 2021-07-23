import cv2
import mediapipe as mp
import time
import pyfakewebcam
import sys

import painting_utils as pu

mp_hands = mp.solutions.hands


def main():

    # check input-output video devices
    if len(sys.argv) < 3:
       raise ValueError("Provide an input video device as first command line argument (eg. '2') and an output video device as second command line argument (eg. '4')")
    print(f"Input video device : {sys.argv[1]}")
    print(f"Output video device : {sys.argv[2]}")

    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
    cap = cv2.VideoCapture(int(sys.argv[1]))
    camera = pyfakewebcam.FakeWebcam('/dev/video' + sys.argv[2], 640, 480)
    painting = set()
    last_position = None

    while cap.isOpened():

        # get camera input
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # execute hand mash
        image = pu.to_readonly_rgb(image, flip=False)
        results = hands.process(image)
        image.flags.writeable = True

        # draw painting
        last_position = pu.handle_painting(image, results, painting, last_position)

        # draw hand annotations (DEBUG)
        if len(sys.argv) == 4 and sys.argv[3] == 'DEBUG':
            pu.draw_hand_annotation(image, results)

        # present image to camera
        camera.schedule_frame(image)
        time.sleep(1/100.0)

    hands.close()
    cap.release()


if __name__ == '__main__':
    main()
