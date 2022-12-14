from djitellopy import Tello
import cv2 as cv
from utils.panorama import Panorama
import time
from utils.cartoon import Cartoon

class TelloKeyboardController:
    def __init__(self, tello: Tello):
        self.tello = tello

    def panorama(self):
        self.tello.rotate_counter_clockwise(25)
        time.sleep(2)
        frame_read = self.tello.get_frame_read()
        cv.imwrite("photo/picture1.jpg", frame_read.frame)

        self.tello.rotate_clockwise(25)
        time.sleep(2)
        frame_read = self.tello.get_frame_read()
        cv.imwrite("photo/picture2.jpg", frame_read.frame)
        
        self.tello.rotate_clockwise(25)
        time.sleep(2)
        frame_read = self.tello.get_frame_read()
        cv.imwrite("photo/picture3.jpg", frame_read.frame)
        self.tello.rotate_counter_clockwise(25)
        Panorama.createPano()

    def cartoon(self):
        frame_read = self.tello.get_frame_read()
        print('cartoon')
        Cartoon.create_cartoon(frame_read.frame)

    def timer(self, count):
        for i in range(count):
            print('taking picture in : ', count-i)
            self.tello.rotate_clockwise(360)
            time.sleep(10)
        frame_read = self.tello.get_frame_read()
        cv.imwrite("photo/timer.jpg", frame_read.frame)
        print('success!')

    def go_self_shoot(self):
        self.tello.rotate_clockwise(180)
        time.sleep(7)
        self.tello.move_forward(100)
        time.sleep(7)
        self.tello.rotate_clockwise(180)
        time.sleep(7)
        frame_read = self.tello.get_frame_read()
        cv.imwrite("photo/go_shoot.jpg", frame_read.frame)
        self.tello.move_forward(100)

    def go_shoot(self):
        self.tello.rotate_clockwise(180)
        time.sleep(7)
        self.tello.move_forward(100)
        time.sleep(7)
        frame_read = self.tello.get_frame_read()
        cv.imwrite("photo/go_shoot.jpg", frame_read.frame)
        time.sleep(1)
        self.tello.rotate_clockwise(180)
        time.sleep(7)
        self.tello.move_forward(100)

    def control(self, key):
        if key == ord('w'):
            self.tello.move_forward(30)
        elif key == ord('s'):
            self.tello.move_back(30)
        elif key == ord('a'):
            self.tello.move_left(30)
        elif key == ord('d'):
            self.tello.move_right(30)
        elif key == ord('e'):
            self.tello.rotate_clockwise(30)
        elif key == ord('q'):
            self.tello.rotate_counter_clockwise(30)
        elif key == ord('r'):
            self.tello.move_up(30)
        elif key == ord('f'):
            self.tello.move_down(30)
        elif key == ord('p'):
            self.panorama()
        elif key == ord('c'):
            self.cartoon()
        elif key == ord('t'):
            self.timer(3)
        elif key == ord('='):
            self.go_self_shoot()
        elif key == ord('-'):
            self.go_shoot()


    



