#!/usr/bin/env python
# -*- coding: utf-8 -*-
import configargparse

import cv2 as cv

from gestures.tello_gesture_controller import TelloGestureController
from utils import CvFpsCalc

from djitellopy import Tello
from gestures import *


import threading

TOLERANCE_X = 5
TOLERANCE_Y = 5
SLOWDOWN_THRESHOLD_X = 20
SLOWDOWN_THRESHOLD_Y = 20
DRONE_SPEED_X = 20
DRONE_SPEED_Y = 20
SET_POINT_X = 960/2
SET_POINT_Y = 720/2


face_cascade = cv.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')

def adjust_tello_position(tello, offset_x, offset_y, offset_z):
    """
    Adjusts the position of the tello drone based on the offset values given from the frame

    :param offset_x: Offset between center and face x coordinates
    :param offset_y: Offset between center and face y coordinates
    :param offset_z: Area of the face detection rectangle on the frame
    """
    if not -90 <= offset_x <= 90 and offset_x is not 0:
        if offset_x < 0:
            tello.rotate_counter_clockwise(10)
        elif offset_x > 0:
            tello.rotate_clockwise(10)
    
    if not -70 <= offset_y <= 70 and offset_y is not -30:
        if offset_y < 0:
            tello.move_up(20)
        elif offset_y > 0:
            tello.move_down(20)
    
    if not 15000 <= offset_z <= 30000 and offset_z is not 0:
        if offset_z < 15000:
            tello.move_forward(20)
        elif offset_z > 30000:
            tello.move_back(20) 

def get_args():
    print('## Reading configuration ##')
    parser = configargparse.ArgParser(default_config_files=['config.txt'])

    parser.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
    parser.add("--device", type=int)
    parser.add("--width", help='cap width', type=int)
    parser.add("--height", help='cap height', type=int)
    parser.add("--is_keyboard", help='To use Keyboard control by default', type=bool)
    parser.add('--use_static_image_mode', action='store_true', help='True if running on photos')
    parser.add("--min_detection_confidence",
               help='min_detection_confidence',
               type=float)
    parser.add("--min_tracking_confidence",
               help='min_tracking_confidence',
               type=float)
    parser.add("--buffer_len",
               help='Length of gesture buffer',
               type=int)

    args = parser.parse_args()

    return args


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def cartoon_filter(img):
    h, w = img.shape[:2]
    img2 = cv.resize(img, (w//2, h//2))

    blr = cv.bilateralFilter(img2, -1, 20, 7)
    edge = 255 - cv.Canny(img2, 80, 120)
    edge = cv.cvtColor(edge, cv.COLOR_GRAY2BGR)
    dst = cv.bitwise_and(blr, edge) # and연산
    dst = cv.resize(dst, (w, h), interpolation=cv.INTER_NEAREST)
                                                                  
    return dst

def main():
    # init global vars
    global gesture_buffer
    global gesture_id
    global battery_status

    # Argument parsing
    args = get_args()
    KEYBOARD_CONTROL = args.is_keyboard
    WRITE_CONTROL = False
    in_flight = False

    # Camera preparation
    tello = Tello()
    tello.connect()
    tello.streamon()

    cap = tello.get_frame_read()

    # Init Tello Controllers
    gesture_controller = TelloGestureController(tello)
    keyboard_controller = TelloKeyboardController(tello)

    gesture_detector = GestureRecognition(args.use_static_image_mode, args.min_detection_confidence,
                                          args.min_tracking_confidence)
    gesture_buffer = GestureBuffer(buffer_len=args.buffer_len)

    def tello_control(key, keyboard_controller, gesture_controller):
        global gesture_buffer

        if KEYBOARD_CONTROL:
            keyboard_controller.control(key)
        else:
            gesture_controller.gesture_control(gesture_buffer)

    def tello_battery(tello):
        global battery_status
        try:
            battery_status = tello.get_battery()[:-2]
        except:
            battery_status = -1

    # FPS Measurement
    cv_fps_calc = CvFpsCalc(buffer_len=10)

    mode = 0
    number = -1
    battery_status = -1

    tello.move_down(20)

    while True:
        fps = cv_fps_calc.get()

        # Process Key (ESC: end)
        key = cv.waitKey(1) & 0xff
        if key == 27:  # ESC
            break
        elif key == 32:  # Space
            if not in_flight:
                # Take-off drone
                tello.takeoff()
                in_flight = True

            elif in_flight:
                # Land tello
                tello.land()
                in_flight = False

        elif key == ord('k'):
            mode = 0
            KEYBOARD_CONTROL = True
            WRITE_CONTROL = False
            tello.send_rc_control(0, 0, 0, 0)  # Stop moving
        elif key == ord('g'):
            KEYBOARD_CONTROL = False
        elif key == ord('n'):
            mode = 1
            WRITE_CONTROL = True
            KEYBOARD_CONTROL = True

        

        if WRITE_CONTROL:
            number = -1
            if 48 <= key <= 57:  # 0 ~ 9
                number = key - 48

        # Camera capture
        image = cap.frame
        # print(image)
        
        debug_image, gesture_id = gesture_detector.recognize(image, number, mode)
        gesture_buffer.add_gesture(gesture_id)

        # Start control thread
        threading.Thread(target=tello_control, args=(key, keyboard_controller, gesture_controller,)).start()
        # threading.Thread(target=tello_battery, args=(tello,)).start()

        debug_image = gesture_detector.draw_info(debug_image, fps, mode, number)

        # Battery status and image rendering
        cv.putText(debug_image, "Battery: {}".format(battery_status), (5, 720 - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # #face detecting
        # gray = cv.cvtColor(debug_image, cv.COLOR_BGR2GRAY)
        # faces = face_cascade.detectMultiScale(  # face detection
        #     gray,
        #     scaleFactor=1.1,
        #     minNeighbors=5,
        #     minSize=(30, 30),
        #     flags=cv.CASCADE_SCALE_IMAGE
        # )

        # i = 0
        # # Decorating image for debug purposes and looping through every detected face
        # for (x, y, w, h) in faces:

        #     cv.rectangle(debug_image, (x, y), (x+w, y+h), (255, 0, 0), 5)  # contour rectangle
        #     cv.circle(debug_image, (int(x+w/2), int(y+h/2)), 12, (255, 0, 0), 1)  # face-centered circle
        #     # print(frame.shape)
        #     # cv2.line(frame, (int(x+w/2), int(720/2)), (int(960/2), int(720/2)), (0, 255, 255))

        #     cv.circle(debug_image, (int(SET_POINT_X), int(SET_POINT_Y)), 12, (255, 255, 0), 8)  # setpoint circle
        #     i = i+1
        #     distanceX = x+w/2 - SET_POINT_X
        #     distanceY = y+h/2 - SET_POINT_Y

        #     up_down_velocity = 0
        #     right_left_velocity = 0

        #     if distanceX < -TOLERANCE_X:
        #         print("sposta il drone alla sua SX")
        #         right_left_velocity = - DRONE_SPEED_X

        #     elif distanceX > TOLERANCE_X:
        #         print("sposta il drone alla sua DX")
        #         right_left_velocity = DRONE_SPEED_X
        #     else:
        #         print("OK")

        #     if distanceY < -TOLERANCE_Y:
        #         print("sposta il drone in ALTO")
        #         up_down_velocity = DRONE_SPEED_Y
        #     elif distanceY > TOLERANCE_Y:
        #         print("sposta il drone in BASSO")
        #         up_down_velocity = - DRONE_SPEED_Y

        #     else:
        #         print("OK")

        #     if abs(distanceX) < SLOWDOWN_THRESHOLD_X:
        #         right_left_velocity = int(right_left_velocity / 2)
        #     if abs(distanceY) < SLOWDOWN_THRESHOLD_Y:
        #         up_down_velocity = int(up_down_velocity / 2)

        #     tello.send_rc_control(right_left_velocity, 0, up_down_velocity, 0)


        cv.imshow('Tello Gesture Recognition', debug_image)

    tello.land()
    tello.end()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
