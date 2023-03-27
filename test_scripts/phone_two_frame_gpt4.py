import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from phone_calibration import calibrate

# camera calibration parameters
K, _, _, _, _, _ = calibrate() 

# Load the two images
path_to_data_folder = '/home/abhilash/Academics/github/visual_odometry/test_scripts/data/android/data'

image1 = cv2.imread(f'{path_to_data_folder}/data_D0.jpeg')
image2 = cv2.imread(f'{path_to_data_folder}/data_A0.jpeg')
# image1 = cv2.imread('image1.jpg')
# image2 = cv2.imread('image2.jpg')

# Create the SIFT object
sift = cv2.SIFT_create()

# Find keypoints and descriptors using SIFT
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# Create a BFMatcher object with the L2 norm
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(descriptors1, descriptors2)

# Sort matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Select the top matches
good_matches = matches[:50]

# Extract matched keypoints
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Calculate the essential matrix
E, mask = cv2.findEssentialMat(src_pts, dst_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

# Decompose the essential matrix into rotation and translation
_, R, t, _ = cv2.recoverPose(E, src_pts, dst_pts, K, mask)

# Print the rotation and translation matrices
print("Rotation matrix (R):\n", R)
print("Translation matrix (t):\n", t)

r = Rotation.from_matrix(R)
angles = r.as_euler("zyx", degrees=True)
print(f"Frame - Rotation:\n{R}\nTranslation:\n{t}\nYaw: {angles[0]} deg, Pitch: {angles[1]} deg, Roll: {angles[2]} deg\n\n")

# Draw the matches, including only the inliers
inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matched features
cv2.imshow('Matched Features', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

