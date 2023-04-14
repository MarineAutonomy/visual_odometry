import cv2
import numpy as np
import matplotlib.pyplot as plt
from justpfm import justpfm
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# function to find depth from the groundtruth pfm
def findDepth(x, y, pfm):
    return np.mean(pfm[y - 3:y + 3, x - 3:x + 3])  # in cm


def getHomoTransMatrix(R, T):
    R_, _ = cv2.Rodrigues(R)
    T_ = np.zeros((4, 4))
    T_[0:3, 0:3] = R_
    T_[0:3, 3] = np.transpose(T)
    T_[3, 0:3] = [0, 0, 0]
    T_[3, 3] = 1
    return T_


noi = 420  # number of data images

# the groundtruth pfm file for the depth data
depth_path = "/home/goldenrishabh/Documents/AbhilashSirGuild/ResearchVision/Code/Data/NewTsukubaStereoDataset" \
             "/groundtruth/depth_maps"

# pfm1 = justpfm.read_pfm(f"{depth_path}/L_000{20}.pfm")
# pfm2 = justpfm.read_pfm(f"{depth_path}/L_000{21}.pfm")

pfm = [justpfm.read_pfm(f"{depth_path}/L_{'{:05d}'.format(i)}.pfm") for i in range(1, noi)]
# intrinsic matrix of the camera
intrinsic = np.array([[615., 0., 320.], [0., 615., 240.], [0., 0., 1.]])

imagePath = "/home/goldenrishabh/Documents/AbhilashSirGuild/ResearchVision/Code/Data/NewTsukubaStereoDataset" \
            "/illumination/daylight"
images = [cv2.imread(f"{imagePath}/L_{'{:05d}'.format(i)}.png") for i in range(1, noi)]

data = pd.read_csv('camera_track.dat', sep=',\s+', header=None, engine='python')
data = data.head(noi)
datafx = []
datafy = []
datafz = []
# print(data[0][4])
# converting images to grayscale
grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in images]

# Feature matching initialization
orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
M_list, dx_list, dy_list = [], [], []
kp_prev, des_prev = orb.detectAndCompute(grays[0], None)
temp = np.zeros((1, 3))

intrinsic_inv = np.linalg.inv(intrinsic)

T_final = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
error = 0
for i in range(0, len(images) - 1):

    # Feature matching
    kp_curr, des_curr = orb.detectAndCompute(grays[i], None)
    matches = bf.match(des_prev, des_curr)
    matches = sorted(matches, key=lambda x: x.distance)
    matches = matches[:50]
    pts_prev = np.float32([kp_prev[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_curr = np.float32([kp_curr[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # extracting the world coordinates for PnP
    object_pts_curr = np.array(
        [np.matmul(intrinsic_inv, findDepth(int(pts[0, 0]), int(pts[0, 1]), pfm[i + 1]) * np.transpose(
            [pts[0, 0], pts[0, 1], 1])) for pts in pts_curr])
    temp = np.array(
        [np.matmul(intrinsic_inv, findDepth(int(pts[0, 0]), int(pts[0, 1]), pfm[i]) * np.transpose(
            [pts[0, 0], pts[0, 1], 1])) for pts in pts_prev])

    # making them homogeneous to multiply by transformation matrix
    object_pts_prev = np.array([[pts[0], pts[1], pts[2], 1] for pts in temp])

    # finding possible error points (depth cannot change drastically between two images)
    idx = np.where((abs(object_pts_prev[:, 2] - object_pts_curr[:, 2]) > 4) == True)

    image_pts_curr = np.array([[pts[0, 0], pts[0, 1]] for pts in pts_curr])
    image_pts_prev = np.array([[pts[0, 0], pts[0, 1]] for pts in pts_prev])

    image_pts_curr = np.delete(image_pts_curr, idx, axis=0)
    image_pts_prev = np.delete(image_pts_prev, idx, axis=0)
    object_pts_prev = np.delete(object_pts_prev, idx, axis=0)
    object_pts_curr = np.delete(object_pts_curr, idx, axis=0)

    object_pts_prev = np.array([np.matmul(np.linalg.inv(T_final), pts) for pts in object_pts_prev])
    object_pts_prev = np.array([[pts[0], pts[1], pts[2]] for pts in object_pts_prev])

    # uncomment to see all the matched performed

    # img3 = cv2.drawMatches(images[0], kp_prev, images[1], kp_curr, matches, None, flags=2)
    # plt.imshow(img3)
    # plt.show(block=False)
    # fig = plt.gcf()
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    # plt.pause(1)

    # Close the plot
    # plt.close()
    # print(object_pts_curr)
    # print(image_pts_prev)

    # print(object_pts_prev-object_pts_curr)
    try:
        _, R1, T1 = cv2.solvePnP(objectPoints=object_pts_prev, imagePoints=image_pts_prev, cameraMatrix=intrinsic,
                                 distCoeffs=np.array([0, 0, 0, 0, 0]))
        _, R2, T2 = cv2.solvePnP(objectPoints=object_pts_prev, imagePoints=image_pts_curr, cameraMatrix=intrinsic,
                                 distCoeffs=np.zeros((1, 5)))
        T_temp = np.matmul(getHomoTransMatrix(R2, T2),np.linalg.inv(getHomoTransMatrix(R1,T1)))
        T_final = np.matmul(T_temp, T_final)
        datafx.append(T_final[0:3, 3][0])
        datafy.append(T_final[0:3, 3][1])
        datafz.append(T_final[0:3, 3][2])
        error += 1
    except:
        print(error)
        error += 1
        continue
    # print(cv2.composeRT(object_pts_prev,object_pts_curr))

    # print(data[1][i] - T_final[0:3, 3][1])
    kp_prev, des_prev = kp_curr, des_curr

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.scatter(datafx, datafy, label='my code')

ax.scatter(data[0], data[1], label='groundtruth')
ax.legend()
plt.show()
