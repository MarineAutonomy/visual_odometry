import cv2
import numpy as np
import matplotlib.pyplot as plt
from justpfm import justpfm


# function to find depth from the groundtruth pfm
def findDepth(x, y, pfm):
     return np.mean(pfm[y - 3:y + 3, x - 3:x + 3])  # in cm
    #return pfm[y,x]

# the groundtruth pfm file for the depth data
depth_path = "/home/goldenrishabh/Documents/AbhilashSirGuild/ResearchVision/Code/Data/NewTsukubaStereoDataset" \
             "/groundtruth/depth_maps"
pfm1 = justpfm.read_pfm(f"{depth_path}/L_0000{1}.pfm")
pfm2 = justpfm.read_pfm(f"{depth_path}/L_000{21}.pfm")

# intrinsic matrix of the camera
intrinsic = np.array([[615., 0., 320.], [0., 615., 240.], [0., 0., 1.]])

imagePath = "/home/goldenrishabh/Documents/AbhilashSirGuild/ResearchVision/Code/Data/NewTsukubaStereoDataset" \
            "/illumination/daylight"
images = [cv2.imread(f"{imagePath}/L_0000{1}.png"),
          cv2.imread(f"{imagePath}/L_000{21}.png")]

# converting images to grayscale
grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# Feature matching initialization
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
M_list, dx_list, dy_list = [], [], []
kp_prev, des_prev = orb.detectAndCompute(grays[0], None)
temp = np.zeros((1, 3))

intrinsic_inv = np.linalg.inv(intrinsic)

for i in range(1, len(images)):
    kp_curr, des_curr = orb.detectAndCompute(grays[i], None)
    matches = bf.match(des_prev, des_curr)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:30]
    pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    temp[0, 0:2] = pts_prev[14][0]
    pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # print([pts[0,0], pts[0,1], 1] for pts in pts_prev)

    object_pts_curr = np.array([np.matmul(intrinsic_inv, findDepth(int(pts[0, 0]), int(pts[0, 1]), pfm2) * np.transpose(
        [pts[0, 0], pts[0, 1], 1])) for pts in pts_curr])
    # print(object_pts_curr)
    object_pts_prev = np.array([ np.matmul(intrinsic_inv, findDepth(int(pts[0, 0]), int(pts[0, 1]), pfm1) * np.transpose(
        [pts[0, 0], pts[0, 1], 1])) for pts in pts_prev])
    idx = np.where( (abs(object_pts_prev[:,2]-object_pts_curr[:,2])>2) == True)

    image_pts_curr = np.array([[pts[0, 0], pts[0, 1]] for pts in pts_curr])
    image_pts_prev = np.array([[pts[0, 0], pts[0, 1]] for pts in pts_prev])

    #print(object_pts_curr-object_pts_prev)
    image_pts_curr = np.delete(image_pts_curr,idx,axis=0)
    image_pts_prev = np.delete(image_pts_prev, idx,axis=0)
    object_pts_prev = np.delete(object_pts_prev, idx,axis=0)
    object_pts_curr = np.delete(object_pts_curr,idx,axis=0)
    img3 = cv2.drawMatches(images[0], kp_prev, images[1], kp_curr, matches, None, flags=2)
    plt.imshow(img3), plt.show()
    #print(object_pts_curr)
    #print(image_pts_prev)

    print(object_pts_prev-object_pts_curr)
    _, R1, T1 = cv2.solvePnP(objectPoints=object_pts_prev, imagePoints=image_pts_prev, cameraMatrix=intrinsic,
                             distCoeffs=np.array([0, 0, 0, 0, 0]))
    _, R2, T2 = cv2.solvePnP(objectPoints=object_pts_prev, imagePoints=image_pts_curr, cameraMatrix=intrinsic,
                             distCoeffs=np.zeros((1, 5)))

    # M_m = cv2.findEssentialMat(pts_curr, pts_prev, intrinsic)
    # M, mask = cv2.findHomography(pts_curr, pts_prev, cv2.RANSAC)
    # dx, dy = M[0, 2], M[1, 2]
    # M_list.append(M)
    # dx_list.append(dx)
    # dy_list.append(dy)

    kp_prev, des_prev = kp_curr, des_curr

# width = sum([img.shape[1] for img in images])
# height = sum([img.shape[0] for img in images])
# result = np.zeros((height, width, 3), dtype=np.uint8)
# x = int(temp[0, 0])
# y = int(temp[0, 1])
# i = 0
# depth = 0


# for pts in pts_prev:
#     depth += findDepth(int((pts[0])[0]), int((pts[0])[1]), pfm1)
#     i += 1

# depth = depth / i
# temp[0, 2] = 1
# temp1 = np.transpose(temp)
# temp2 = np.matmul(M_list[0], temp1)
# temp2[2, 0] = 1
# temp1_c = np.matmul(np.linalg.inv(intrinsic), temp1) * depth
# temp2_c = np.matmul(np.linalg.inv(intrinsic), temp2) * depth
# final = temp2_c - temp1_c
# print(M_m[0])
# num, Rs, Ts, Ns = cv2.decomposeHomographyMat(M_list[0], intrinsic)
#print(T1)
#print(T2)
R1_,_ = cv2.Rodrigues(R1)
R2_,_ = cv2.Rodrigues(R2)
T1_ = np.zeros((4,4))
T1_[0:3,0:3] = R1_
T1_[0:3,3] = np.transpose(T1)
T1_[3,0:3] = [0,0,0]
T1_[3,3] = 1

T2_ = np.zeros((4,4))
T2_[0:3,0:3] = R2_
T2_[0:3,3] = np.transpose(T2)
T2_[3,0:3] = [0,0,0]
T2_[3,3] = 1

print(np.matmul(T2_,np.linalg.inv(T1_)))
# print(object_pts_curr-object_pts_prev)
# print(f"You have moved {final[0, 0]}m in x-direction {final[1, 0]}m in y direction {final[2, 0]}m in z direction")

# cv2.imshow('Panorama', result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
