import cv2
import numpy as np

images = [cv2.imread(f'./test_images/image{i}.jpeg') for i in range(1, 2)]
grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
M_list, dx_list, dy_list = [], [], []
kp_prev, des_prev = orb.detectAndCompute(grays[0], None)

for i in range(1, len(images)):
    kp_curr, des_curr = orb.detectAndCompute(grays[i], None)
    matches = bf.match(des_prev, des_curr)
    matches = sorted(matches, key=lambda x:x.distance)
    matches = matches[:50]
    pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(pts_curr, pts_prev, cv2.RANSAC)
    dx, dy = M[0, 2], M[1, 2]
    M_list.append(M)
    dx_list.append(dx)
    dy_list.append(dy)
    kp_prev, des_prev = kp_curr, des_curr

width = sum([img.shape[1] for img in images])
height = max([img.shape[0] for img in images])
result = np.zeros((height, width, 3), dtype=np.uint8)
x = 0

for i, img in enumerate(images):
    if i == 0:
        warped = img
    else:
        M_prev = M_list[i-1]
        warped = cv2.warpPerspective(img, M_prev, (img.shape[1] + int(abs(dx_list[i-1])), img.shape[0] + int(abs(dy_list[i-1]))))
    dx = M[0, 2]
    dy = M[1, 2]
    if dx >= 0 and dy >= 0:
        result[int(abs(dy)):int(abs(dy))+warped.shape[0], x:x+warped.shape[1]] = warped
    elif dx < 0 and dy >= 0:
        result[int(abs(dy)):int(abs(dy))+warped.shape[0], x+int(abs(dx)):x+warped.shape[1]] = warped[:, -int(abs(dx)):, :]
    elif dx >= 0 and dy < 0:
        result[0:warped.shape[0]+int(abs(dy)), x:x+warped.shape[1]] = warped[-int(abs(dy)):, :, :]
    else:
        result[0:warped.shape[0]+int(abs(dy)), x+int(abs(dx)):x+warped.shape[1]+int(abs(dx))] = warped[-int(abs(dy)):, -int(abs(dx)):, :]
    x += img.shape[1]

cv2.imshow('Panorama', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
