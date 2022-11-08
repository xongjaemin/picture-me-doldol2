#!/usr/bin/env python
# -*- coding: utf-8 -*-
import configargparse

import cv2 as cv

from gestures.tello_gesture_controller import TelloGestureController
from utils import CvFpsCalc

from djitellopy import Tello
from gestures import *


import threading


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
            tello.move_backward(20) 

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
        


        #face detecting
        """
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        """
        height, width, _ = image.shape

        # Calculate frame center
        center_x = int(width/2)
        center_y = int(height/2)

        # Draw the center of the frame
        cv.circle(debug_image, (center_x, center_y), 10, (0, 255, 0))

        # Convert frame to grayscale in order to apply the haar cascade
        gray = cv.cvtColor(debug_image, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5)

        # If a face is recognized, draw a rectangle over it and add it to the face list
        face_center_x = center_x
        face_center_y = center_y
        z_area = 0
        for face in faces:
            (x, y, w, h) = face
            cv.rectangle(debug_image,(x, y),(x + w, y + h),(255, 255, 0), 2)

            face_center_x = x + int(h/2)
            face_center_y = y + int(w/2)
            z_area = w * h

            cv.circle(debug_image, (face_center_x, face_center_y), 10, (0, 0, 255))

        # Calculate recognized face offset from center
        offset_x = face_center_x - center_x
        # Add 30 so that the drone covers as much of the subject as possible
        offset_y = face_center_y - center_y - 30

        cv.putText(debug_image, f'[{offset_x}, {offset_y}, {z_area}]', (10, 50), cv.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv.LINE_AA)
        # adjust_tello_position(tello, offset_x, offset_y, z_area)

        cv.imshow('Tello Gesture Recognition', debug_image)

    tello.land()
    tello.end()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
