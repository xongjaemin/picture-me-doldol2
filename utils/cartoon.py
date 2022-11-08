import cv2
import sys
import time
import numpy as np

class Cartoon:

  def cartoon_filter(img):
      h, w = img.shape[:2]
      img2 = cv2.resize(img, (w//2, h//2))

      blr = cv2.bilateralFilter(img2, -1, 20, 7)
      edge = 255 - cv2.Canny(img2, 80, 120)
      edge = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
      dst = cv2.bitwise_and(blr, edge) # and연산
      dst = cv2.resize(dst, (w, h), interpolation=cv2.INTER_NEAREST)
                                                                    
      return dst

  '''
  ref : https://towardsdatascience.com/turn-photos-into-cartoons-using-python-bb1a9f578a7e
  '''

  def color_quantization(img, k):
  # Transform the image
    data = np.float32(img).reshape((-1, 3))

  # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

  # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result

  def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
    return edges

  def create_cartoon(drone_capture):
    #edge
    line_size = 7
    blur_value = 7
    edges = Cartoon.edge_mask(drone_capture, line_size, blur_value)
    cv2.imwrite('photo/cartoon_edge.jpg', edges)

    #color
    total_color = 9
    img = Cartoon.color_quantization(drone_capture, total_color)
    cv2.imwrite('photo/cartoon_color.jpg',img)
    
    #blurred
    blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200,sigmaSpace=200)
    cv2.imwrite('photo/cartoon_blur.jpg',blurred)

    #cartoon
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    cv2.imwrite('photo/cartoon_result.jpg',cartoon)
    print('success!')



'''
main code

cap = cv2.VideoCapture(0) # 카메라 오픈.

if not cap.isOpened():
    print('video open failed!')
    sys.exit()
    
while True:                 # 무한 루프
    ret, frame = cap.read() # 웹 카메라의 프레임값 불러오기
    
    if not ret:
        break
    
    # frame = cartoon_filter(frame) # 프레임에 카툰 필터 적용

    # cv2.imshow('frame',frame)
    # cv2.imwrite('photo/cartoon.jpg', frame)
    # key = cv2.waitKey(1) # 다음 프레임을 위해서 빠르게 1ms 간격으로 전환
    
    # if key == 27: # esc 누르면 종료
    #     break

    #edge
    line_size = 7
    blur_value = 7
    edges = edge_mask(frame, line_size, blur_value)
    cv2.imwrite('photo/cartoon_edge.jpg', edges)

    #color
    total_color = 9
    img = color_quantization(frame, total_color)
    cv2.imwrite('photo/cartoon_color.jpg',img)
    
    #blurred
    blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200,sigmaSpace=200)
    cv2.imwrite('photo/cartoon_blur.jpg',blurred)

    #cartoon
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    cv2.imwrite('photo/cartoon_result.jpg',cartoon)

    #show screen
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) # 다음 프레임을 위해서 빠르게 1ms 간격으로 전환
    
    if key == 27: # esc 누르면 종료
        break



cap.release()
cv2.destroyAllWindows()
'''