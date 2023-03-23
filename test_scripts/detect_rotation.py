import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from numpy import set_printoptions, arange
set_printoptions(precision=5, threshold=5, edgeitems=4, suppress=True)

img1 = cv2.imread("/home/vallabh/visual_odometry/tutorials/data/vo_test/image1.jpg", cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread("/home/vallabh/visual_odometry/tutorials/data/vo_test/image1.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("/home/vallabh/visual_odometry/tutorials/data/vo_test/image2.jpg", cv2.IMREAD_GRAYSCALE)

# ORB Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# SIFT
# sift = cv2.xfeatures2d.SIFT_create()
# kp1, des1 = sift.detectAndCompute(img1, None)
# kp2, des2 = sift.detectAndCompute(img2, None)

# for d in des1:
#     print(d)

# for kp in kp1:
#     print(kp)

# Brute Force Matching (NORM_L1 for SIFT, NORM_HAMMING for ORB)
# bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)

# print(len(matches))
# for m in matches:
#     print(m.distance)

list_kp1 = [kp1[mat.queryIdx].pt for mat in matches] 
list_kp2 = [kp2[mat.trainIdx].pt for mat in matches]

k = np.array([[3080.509033203125, 0.0, 1909.4766845703125], [0.0, 3080.509033203125, 1122.2470703125], [0.0, 0.0, 1.0]])
# n = len(matches)
n = 8
print("n", n)
CMat = np.zeros((n,9))

good_new = []
good_old = []

def _form_transf(R, t):
    """
    Makes a transformation matrix from the given rotation matrix and translation vector
    Parameters
    ----------
    R (ndarray): The rotation matrix
    t (list): The translation vector
    Returns
    -------
    T (ndarray): The transformation matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.T
    return T

def sum_z_cal_relative_scale(R, t, q1, q2):
    # Get the transformation matrix
    T_new = _form_transf(R, t)
    # Make the projection matrix
    P_new = np.matmul(np.concatenate((k*1.55e-6, np.zeros((3, 1))), axis=1), T_new)
    T_old = _form_transf(np.eye(3), np.zeros((3, 1)))
    P_old = np.matmul(np.concatenate((k*1.55e-6, np.zeros((3, 1))), axis=1), T_old)

    # Triangulate the 3D points
    hom_Q1 = cv2.triangulatePoints(P_old, P_new, q1.T, q2.T)

    # Also seen from cam 2
    hom_Q2 = np.matmul(T_new@T_old, hom_Q1)

    # Un-homogenize
    uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
    uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

    # print(uhom_Q1)
    # print(uhom_Q1.T[:-1])
    # print(np.linalg.norm(uhom_Q1.T[:-1] - np.flip(uhom_Q1.T[1:], axis=0), axis=-1)/np.linalg.norm(uhom_Q2.T[:-1] - np.flip(uhom_Q2.T[1:], axis=0), axis=-1))

    # Find the number of points there has positive z coordinate in both cameras
    sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
    sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

    # Form point pairs and calculate the relative scale
    relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - np.flip(uhom_Q1.T[1:], axis=0), axis=-1)/
                                np.linalg.norm(uhom_Q2.T[:-1] - np.flip(uhom_Q2.T[1:], axis=0), axis=-1))
    
    # print(relative_scale)
    
    return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale

for i in range(n):
    u1 = list_kp1[i][0]
    v1 = list_kp1[i][1]
    u2 = list_kp2[i][0]
    v2 = list_kp2[i][1]
    good_old.append([u1,v1])
    good_new.append([u2,v2])
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

W1 = np.array([[0,1,0],[-1,0,0],[0,0,1]])
W2 = np.array([[0,-1,0],[1,0,0],[0,0,1]])

R1 = U @ -W1 @ V
r =  Rotation.from_matrix(R1)
angles = r.as_euler("zyx",degrees=True)
# print("R1", angles)

R2 = U @ -W2 @ V
r =  Rotation.from_matrix(R2)
angles = r.as_euler("zyx",degrees=True)
# print("R2", angles)

E, _ = cv2.findEssentialMat(np.array(good_old), np.array(good_new), k, cv2.RANSAC, 0.999, 1.0)
_, R, t, _ = cv2.recoverPose(E, np.array(good_old), np.array(good_new), k)
z_sum, scale = sum_z_cal_relative_scale(R, t, np.array(good_old), np.array(good_new))
t = t*scale
print("E", E)
print("R", R)
print("t", t)
r =  Rotation.from_matrix(R)
angles = r.as_euler("zyx",degrees=True)
print("Rotation angles", angles)
# print("SVD", E - (Ue @ np.diag(Se) @ Ve))




def skew_sym_vec(t):
    tx = np.array([[0,-t[2,0],t[1,0]],[t[2,0],0,-t[0,0]],[t[1,0],t[0,0],0]])
    return tx


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
