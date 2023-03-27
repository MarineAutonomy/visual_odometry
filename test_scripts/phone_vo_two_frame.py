import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from phone_calibration import calibrate

def decompose_essential_matrix(E):
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R = np.dot(np.dot(U, W), Vt)
    t = U[:, 2]

    # Ensure R is a proper rotation matrix
    if np.linalg.det(R) < 0:
        R = -R

    return R, t


def triangulate_points(pts1, pts2, P1, P2):
    triangulated_points = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    pts_3d = cv2.convertPointsFromHomogeneous(triangulated_points.T)[:, 0, :]
    
    # Filter out points with invalid depths (Z-coordinates)
    min_depth = 0.1  # Adjust this value as needed
    max_depth = 10  # Adjust this value as needed
    
    valid_indices = np.where((pts_3d[:, 2] > min_depth) & (pts_3d[:, 2] < max_depth))[0]
    
    filtered_pts_3d = pts_3d[valid_indices]
    
    return filtered_pts_3d

# camera calibration parameters
K, _, _, _, _, _ = calibrate() 
print(K)

# initialize variables
R = np.eye(3)
t = np.zeros((3, 1))

# feature detector and descriptor
sift = cv2.SIFT_create()
# surf = cv2.xfeatures2d.SURF_create()

path_to_data_folder = '/home/abhilash/Academics/github/visual_odometry/test_scripts/data/android/data'

prev_frame = cv2.imread(f'{path_to_data_folder}/data_00.jpeg')
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

prev_kps, prev_descs = sift.detectAndCompute(prev_gray, None)
# prev_kps, prev_descs = surf.detectAndCompute(prev_gray, None)

prev_pts = np.array([kp.pt for kp in prev_kps], dtype=np.float32).reshape(-1, 1, 2)

curr_frame = cv2.imread(f'{path_to_data_folder}/data_01.jpeg')
curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
curr_kps, curr_descs = sift.detectAndCompute(curr_gray, None)
# curr_kps, curr_descs = surf.detectAndCompute(curr_gray, None)


# match features between frames
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(prev_descs, curr_descs)
matches = sorted(matches, key=lambda x: x.distance)

# Select the top matches
good_matches = matches[:50]

# estimate motion from feature matches
src_pts = np.array([prev_kps[m.queryIdx] for m in good_matches]).reshape(-1,1,2)
dst_pts = np.array([curr_kps[m.trainIdx] for m in good_matches]).reshape(-1,1,2)

# Estimate essential matrix
E, mask = cv2.findEssentialMat(src_pts, dst_pts, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
if E is None:
    print(f'Essential matrix could not be computed')

# Recover the pose
_, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K, mask=mask)


r = Rotation.from_matrix(R)
angles = r.as_euler("zyx", degrees=True)
print(f"Frame - Rotation:\n{R}\nTranslation:\n{t}\nYaw: {angles[0]} deg, Pitch: {angles[1]} deg, Roll: {angles[2]} deg\n\n")

inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
vis_features = cv2.drawMatches(prev_gray, prev_kps, curr_gray, curr_kps, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("Visual Odometry", vis_features)

# check for key press to exit
cv2.waitKey(0)
cv2.destroyAllWindows()