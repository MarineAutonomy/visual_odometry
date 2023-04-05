
# import OpenCV and pyplot 
import cv2 
from matplotlib import pyplot as plt
  
# read left and right images
imgA = cv2.imread('left 1A0.png',0)
imgB = cv2.imread('right 1A0.png',0)
cv2.imshow(imgA,'imgA')
imgL = cv2.resize(imgA, (640, 480))
imgR = cv2.resize(imgB, (640, 480))

# creates StereoBm object 
stereo = cv2.StereoBM_create(numDisparities = 16,
                            blockSize = 15)
  
# computes disparity
disparity = stereo.compute(imgL, imgR)
  
# displays image as grayscale and plotted
plt.imshow(disparity, 'gray')
plt.show()
