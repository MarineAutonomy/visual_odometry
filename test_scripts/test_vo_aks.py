
# import OpenCV and pyplot 
import cv2 
from matplotlib import pyplot as plt
  
# read left and right images
imgA = cv2.imread('/home/akash/visual_odometry/test_scripts/test_Apr05_2023/left_A0.jpg',0)
imgB = cv2.imread('/home/akash/visual_odometry/test_scripts/test_Apr05_2023/right_A0.jpg',0)
#print(imgA.shape())

imgL = cv2.resize(imgA, (640, 480))
imgR = cv2.resize(imgB, (640, 480))

# creates StereoBm object 
stereo = cv2.StereoBM_create(numDisparities = 16,
                            blockSize = 15)
  
# computes disparity
disparity = stereo.compute(imgL, imgR)
cv2.imshow('imgA',imgA)
cv2.imshow('imgB',imgB)
# displays image as grayscale and plotted
plt.imshow(disparity, 'gray')
plt.show()
