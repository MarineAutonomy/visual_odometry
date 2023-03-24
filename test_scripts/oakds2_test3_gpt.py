# In this updated version, we've made the following improvements:

# Replaced optical flow with a feature matcher (Brute-Force Matcher) for feature matching.
# Updated the visual odometry loop to use matched keypoints instead of optical flow results.
# Visualized feature matches using cv2.drawMatches() instead of drawing motion vectors.
# These changes should provide more accurate and robust visual odometry results.
#
# This updated function filters out 3D points with Z-coordinates outside the specified 
# range (min_depth to max_depth). You can adjust these values according to your specific 
# requirements or the environment in which your application operates.

import cv2
import depthai as dai
import numpy as np
from scipy.spatial.transform import Rotation
import sys, os


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
    # print('Hello 1')
    # Convert points to homogeneous form
    # pts1_homogeneous = cv2.convertPointsToHomogeneous(pts1)[:, 0, :]
    # pts2_homogeneous = cv2.convertPointsToHomogeneous(pts2)[:, 0, :]
    # print('Hello 2', np.shape(P1), np.shape(P2), np.shape(pts1), np.shape(pts2))
    triangulated_points = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    # print('Hello 3')
    pts_3d = cv2.convertPointsFromHomogeneous(triangulated_points.T)[:, 0, :]
    # print('Hello 3a', np.shape(pts_3d))
    # Filter out points with invalid depths (Z-coordinates)
    min_depth = 0.1  # Adjust this value as needed
    max_depth = 10  # Adjust this value as needed
    
    valid_indices = np.where((pts_3d[:, 2] > min_depth) & (pts_3d[:, 2] < max_depth))[0]
    # print('Hello 4')
    filtered_pts_3d = pts_3d[valid_indices]
    # print('Hello 5')
    return filtered_pts_3d

MIN_NUMBER_OF_FEATURES = 100

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output
cam_rgb = pipeline.createColorCamera()
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")

# Configure properties
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam_rgb.setFps(30)

# Link the nodes
cam_rgb.preview.link(xout_rgb.input)

# Connect to device and start the pipeline
device = dai.Device()
device.startPipeline(pipeline)

# Print intrinsic data of RGB camera
calibData = device.readCalibration()
K_rgb, _, _ = calibData.getDefaultIntrinsics(dai.CameraBoardSocket.RGB)
K_rgb = np.array(K_rgb)
print("RGB Camera Default intrinsics...")
print(K_rgb)

# Output stream
q_rgb = device.getOutputQueue(name="rgb", maxSize=2, blocking=False)

# Create feature detector
orb = cv2.ORB_create(nfeatures=1000)
sift = cv2.xfeatures2d.SIFT_create()

# Create a feature matcher (Brute-Force Matcher)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Initialize previous frame and points
prev_frame = None
prev_keypoints = None
prev_descriptors = None

# Initialize camera pose
R_total = np.eye(3)
t_total = np.zeros((3, 1))

cumul_R = R_total.copy()

frame_count = 0
frame_flag = 0
while True:
    # Get the image from the rgb queue
    in_rgb = q_rgb.tryGet()

    if in_rgb is not None:
        # Convert image to grayscale
        frame = in_rgb.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect features in the first frame
        if prev_frame is None:
            prev_frame = gray

            prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_frame, None)
            # prev_keypoints, prev_descriptors = sift.detectAndCompute(prev_frame, None)
        else:
            try:
                good_matches = []
                keypoints, descriptors = orb.detectAndCompute(gray, None)
                # keypoints, descriptors = sift.detectAndCompute(gray, None)

                if np.shape(prev_keypoints)[0] < MIN_NUMBER_OF_FEATURES:
                    prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_frame, None)
                    # prev_keypoints, prev_descriptors = sift.detectAndCompute(prev_frame, None)
                    print(np.shape(prev_descriptors), np.shape(descriptors))
                
                
                # Match features between frames
                matches = bf.match(prev_descriptors, descriptors)
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Filter out top N matches
                good_matches = matches[:MIN_NUMBER_OF_FEATURES]
                
                # Get matched points
                src_pts = np.float32([prev_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                # Estimate camera motion using essential matrix
                E, mask = cv2.findEssentialMat(dst_pts, src_pts, cameraMatrix=K_rgb, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                # R, t = decompose_essential_matrix(E)
                retval, R, t, mask = cv2.recoverPose(E, dst_pts, src_pts, cameraMatrix=K_rgb, mask=mask)

                print(np.linalg.det(R), np.linalg.norm(t))

                if np.linalg.det(R) < 0:
                    R *= -1
                    t *= -1
                # print(np.shape(t))



                # print(R, t)

                #----------------------------------------------------
                # Triangulate points to estimate scale
                P1 = np.dot(K_rgb, np.hstack((R_total, t_total)))
                P2 = np.dot(K_rgb, np.hstack((np.dot(R_total, R), t_total + np.dot(R_total, t))))
                pts_3d = triangulate_points(src_pts, dst_pts, P1, P2)

                if pts_3d.size > 0:
                    # Compute relative scale
                    scale = np.median(pts_3d[:, 2])

                    # Update camera pose
                    t_total += scale * np.dot(R_total, t) #[:, np.newaxis]
                    R_total = np.dot(R_total, R)

                    cumul_R = np.dot(cumul_R, R)
                    r = Rotation.from_matrix(cumul_R)
                    angles = r.as_euler("zyx", degrees=True)

                    print(f"Frame {frame_count+1} - Rotation:\n{R_total}\nTranslation:\n{t_total}\nYaw: {angles[0]} deg, Pitch: {angles[1]} deg, Roll: {angles[2]} deg\n\n")
                else:
                    print(f"Frame {frame_count+1} - No valid triangulated points")

                #-----------------------------------------------------
                
                # # Triangulate points to estimate scale
                # P1 = np.dot(K_rgb, np.hstack((R_total, t_total)))
                # P2 = np.dot(K_rgb, np.hstack((np.dot(R_total, R), t_total + np.dot(R_total, t))))
                # pts_3d = triangulate_points(src_pts, dst_pts, P1, P2)

                # # Compute relative scale
                # scale = np.median(pts_3d[:, :, 2])

                # # Update camera pose
                # t_total += scale * np.dot(R_total, t)
                # R_total = np.dot(R_total, R)

                # cumul_R = np.dot(cumul_R, R)
                # r = Rotation.from_matrix(cumul_R)
                # angles = r.as_euler("zyx", degrees=True)

                # print(f"Frame {frame_count+1} - Rotation:\n{R_total}\nTranslation:\n{t_total}\nYaw: {angles[0]} deg, Pitch: {angles[1]} deg, Roll: {angles[2]} deg\n\n")

                # Update previous keypoints, descriptors, and frame
                prev_keypoints = keypoints
                prev_descriptors = descriptors
                prev_frame = gray

            except Exception as e:
                # print(f'{frame_count} : Error in Visual Odometry - {e}')                
                frame_flag = 1
                prev_keypoints = keypoints
                prev_descriptors = descriptors
                prev_frame = gray

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                
                print(exc_type, fname, exc_tb.tb_lineno)
                
                pass

            # Visualize feature matches
            # vis_features = cv2.drawKeypoints(gray, keypoints, frame)
            vis_features = cv2.drawMatches(prev_frame, prev_keypoints, frame, keypoints, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Visual Odometry", vis_features)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

    frame_count += 1
    frame_flag = 0

# Cleanup
cv2.destroyAllWindows()



