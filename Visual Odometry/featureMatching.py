import cv2
import numpy as np
import matplotlib.pyplot as plt

intrinsic = np.array([[608.76911527, 0, 331.66232379],
                      [0, 609.17361504, 235.85464047],
                      [0, 0, 1]])
images = [cv2.imread(f'./test_images/image{i}.jpeg') for i in range(3,5)]
grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
M_list, dx_list, dy_list = [], [], []
kp_prev, des_prev = orb.detectAndCompute(grays[0], None)
#h,w = [sum(img.shape[1] for img in images),sum(img.shape[0] for img in images)]
temp = np.zeros((1,3))

for i in range(1, len(images)):
    kp_curr, des_curr = orb.detectAndCompute(grays[i], None)
    matches = bf.match(des_prev, des_curr)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:50]
    pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    temp[0,0:2] = pts_prev[30][0]
    pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(pts_curr, pts_prev, cv2.RANSAC)
    dx, dy = M[0, 2], M[1, 2]
    M_list.append(M)
    dx_list.append(dx)
    dy_list.append(dy)
    img3 = cv2.drawMatches(images[0], kp_prev, images[1], kp_curr, matches,None, flags=2)
    plt.imshow(img3), plt.show()
    kp_prev, des_prev = kp_curr, des_curr


width = sum([img.shape[1] for img in images])
height = sum([img.shape[0] for img in images])
result = np.zeros((height, width, 3), dtype=np.uint8)
x = 0

temp[0,2] = 1
temp1 = np.transpose(temp)
temp2 = np.matmul(M_list[0],temp1)
temp2[2,0] = 1
temp1_c = np.matmul(np.linalg.inv(intrinsic),temp1)*2.5
temp2_c = np.matmul(np.linalg.inv(intrinsic),temp2)*2.5

final = temp2_c-temp1_c

print(f"You have moved {final[0,0]*3.2}ft in x-direction {final[1,0]*3.2}ft in y direction {final[2,0]*3.2}ft in z direction")
# cv2.imshow('Panorama', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
