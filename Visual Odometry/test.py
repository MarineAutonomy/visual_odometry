import cv2
import numpy as np
img1 = cv2.imread('./Depth-Rectification-on-host-static/L_00001.png')
img2 = cv2.imread('./Depth-Rectification-on-host-static/R_00001.png')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)
bf = cv2.BFMatcher()
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
num_matches = int(len(matches) * 0.1)
matches = matches[:num_matches]
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC, 0.1, 0.99)
E, _ = cv2.findEssentialMat(src_pts, dst_pts)
K = np.array([[1000, 0, img1.shape[1]/2], [0, 1000, img1.shape[0]/2], [0, 0, 1]])
R1, R2, T = cv2.decomposeEssentialMat(E)
P1 = np.hstack((K, np.zeros((3, 1))))
P2_1 = np.hstack((np.dot(K, R1), T))
P2_2 = np.hstack((np.dot(K, R1), -T))
P2_3 = np.hstack((np.dot(K, R2), T))
P2_4 = np.hstack((np.dot(K, R2), -T))
points4D_1 = cv2.triangulatePoints(P1, P2_1, src_pts, dst_pts)
points4D_2 = cv2.triangulatePoints(P1, P2_2, src_pts, dst_pts)
points4D_3 = cv2.triangulatePoints(P1, P2_3, src_pts, dst_pts)
points4D_4 = cv2.triangulatePoints(P1, P2_4, src_pts, dst_pts)
points3D_1 = cv2.convertPointsFromHomogeneous(points4D_1.T)[:, 0, :]
points3D_2 = cv2.convertPointsFromHomogeneous(points4D_2.T)[:, 0, :]
points3D_3 = cv2.convertPointsFromHomogeneous(points4D_3.T)[:, 0, :]
points3D_4 = cv2.convertPointsFromHomogeneous(points4D_4.T)[:, 0, :]
