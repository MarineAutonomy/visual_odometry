import cv2
import numpy as np
from numpy import set_printoptions, arange
set_printoptions(precision=5, threshold=5, edgeitems=4, suppress=True)

# img1 = cv2.imread("/home/vallabh/visual_odometry/tutorials/data/feature_detection_test/along_edge.jpg", cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("/home/vallabh/visual_odometry/tutorials/data/feature_detection_test/perpendicular_to_edge.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/home/vallabh/visual_odometry/tutorials/data/feature_detection_test/perpendicular_to_edge.jpg", cv2.IMREAD_GRAYSCALE)

# ORB Detector
# orb = cv2.ORB_create()
# kp1, des1 = orb.detectAndCompute(img1, None)
# kp2, des2 = orb.detectAndCompute(img2, None)

# SIFT
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# for d in des1:
#     print(d)

# for kp in kp1:
#     print(kp)

# Brute Force Matching
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)

# print(len(matches))
# for m in matches:
#     print(m.distance)

list_kp1 = [kp1[mat.queryIdx].pt for mat in matches] 
list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

k = np.array([[3114.12451171875, 0.0, 1910.6363525390625], [0.0, 3114.12451171875, 1076.145751953125], [0.0, 0.0, 1.0]])
n = len(matches)
print("n", n)
CMat = np.zeros((n,9))

for i in range(n):
    u1 = int(list_kp1[i][0])
    v1 = int(list_kp1[i][1])
    u2 = int(list_kp2[i][0])
    v2 = int(list_kp2[i][1])
    p1 = np.linalg.inv(k) @ [u1, v1, 1]
    # print("p1", p1)
    p2 = np.linalg.inv(k) @ [u2, v2, 1]
    # print("p2", p2)
    p1 = p1/p1[2]
    p2 = p2/p2[2]
    CMat[i] = [p2[0]*p1[0], p2[0]*p1[1], p2[0], p2[1]*p1[0], p2[1]*p1[1], p2[1], p1[0], p1[1], 1]

U, S, V = np.linalg.svd(CMat, full_matrices=True)
f = V[-1, :]
E = f.reshape(3, 3)

U, S, V = np.linalg.svd(E, full_matrices=True)

# zero out the last singular value
# project the E matrix onto space of valid essential matrices
S = [1, 1, 0]
E = U @ np.diag(S) @ V

# print("SVD", E - (Ue @ np.diag(Se) @ Ve))

Wt1 = np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]])
Wt2 = np.array([[0, -1, 0],[1, 0, 0],[0, 0, 1]])
R1 = U @ Wt1 @ V
R2 = U @ Wt2 @ V
R3 = U @ -Wt1 @ V
R4 = U @ -Wt2 @ V

vec = [1, 0, 0]

print("R1", (180/np.pi)*np.arccos(np.dot(vec, R1 @ vec)))
print("R2", (180/np.pi)*np.arccos(np.dot(vec, R2 @ vec)))
print("R3", (180/np.pi)*np.arccos(np.dot(vec, R3 @ vec)))
print("R4", (180/np.pi)*np.arccos(np.dot(vec, R4 @ vec)))

# print(list_kp1[0])
# print(list_kp1[0][0])
# print(list_kp1[0][1])
# print(list_kp2[0])

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:n], img2, flags=2)

img1 = cv2.circle(img1, (int(list_kp1[0][0]), int(list_kp1[0][1])), 2, (255,255,128), 2)
img2 = cv2.circle(img2, (int(list_kp2[0][0]), int(list_kp2[0][1])), 2, (255,255,128), 2)

img1 = cv2.circle(img1, (int(list_kp1[1][0]), int(list_kp1[1][1])), 2, (255,255,128), 2)
img2 = cv2.circle(img2, (int(list_kp2[1][0]), int(list_kp2[1][1])), 2, (255,255,128), 2)

while True:
    cv2.imshow("Img1", img1)
    cv2.imshow("Img2", img2)
    cv2.imshow("Img3", img3)
    if cv2.waitKey(1) == ord('q'):
        break
