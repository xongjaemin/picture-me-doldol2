import cv2
import numpy as np
import matplotlib.pyplot as plt

class Panorama:
        def createPano():
            img_list = ['photo/picture1.jpg', 'photo/picture2.jpg', 'photo/picture3.jpg']
            img_list = sorted(img_list)

            print(img_list)
            imgs = []

            for i, img_path in enumerate(img_list):
                img = cv2.imread(img_path)
                imgs.append(img)

            mode = cv2.STITCHER_PANORAMA
            # mode = cv2.STITCHER_SCANS

            if int(cv2.__version__[0]) == 3:
                stitcher = cv2.createStitcher(mode)
            else:
                stitcher = cv2.Stitcher_create(mode)
                
            status, stitched = stitcher.stitch(imgs)

            if status == 0:
                cv2.imwrite('photo/result.jpg', stitched)

                plt.figure(figsize=(20, 20))
                plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
            else:
                print('failed... %s' % status)

            gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
            thresh = cv2.bitwise_not(cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1])
            thresh = cv2.medianBlur(thresh, 5)

            stitched_copy = stitched.copy()
            thresh_copy = thresh.copy()

            while np.sum(thresh_copy) > 0:
                thresh_copy = thresh_copy[1:-1, 1:-1]
                stitched_copy = stitched_copy[1:-1, 1:-1]
                
            cv2.imwrite('photo/result_crop.jpg',stitched_copy)


Panorama.createPano()