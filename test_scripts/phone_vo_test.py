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

# set up video stream from USB camera
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('http://192.168.0.100:8080/video')
cap = cv2.VideoCapture('http://0.0.0.0:4747/mjpegfeed?')

# camera calibration parameters
K, _, _, _, _, _ = calibrate() 
print(K)

# initialize variables
prev_frame = None
prev_kps = None
prev_descs = None
R = np.eye(3)
t = np.zeros((3, 1))
R_total = np.eye(3)
t_total = np.zeros((3, 1))

# feature detector and descriptor
sift = cv2.SIFT_create()

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_frame = gray.copy()
prev_kps, prev_descs = sift.detectAndCompute(gray, None)
prev_pts = np.array([kp.pt for kp in prev_kps], dtype=np.float32).reshape(-1, 1, 2)

fps = cap.get(cv2.CAP_PROP_FPS)
print(f'FPS: {fps}')

# loop over frames from video stream
frame_count = 0
while True:
    # read frame from video stream
    ret, frame = cap.read()    

    if frame_count % 10 == 0:
    
        # undistort frame using camera calibration parameters
        # frame = cv2.undistort(frame, K, None)

        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # track features in current frame
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, gray, prev_pts, None)
        status_flat = np.array([item for sublist in status for item in sublist])

        # print(curr_pts, status)

        # select only valid points
        good_curr_pts = curr_pts[status_flat == 1, 0, :]
        good_prev_pts = prev_pts[status_flat == 1, 0, :]

        # select keypoints for matching and find their descriptors
        good_prev_kps = [cv2.KeyPoint(x[0], x[1], 1) for x in good_prev_pts]
        good_curr_kps = [cv2.KeyPoint(x[0], x[1], 1) for x in good_curr_pts]
        good_curr_descs = sift.compute(gray, good_curr_kps)[1]

        # select previous descriptors for matching
        
        good_prev_descs = prev_descs[status_flat == 1, :]
        # print(np.shape(good_prev_descs))
        

        # print('Hello 2')
        # match features between frames
        bf = cv2.BFMatcher()
        matches = bf.match(good_prev_descs, good_curr_descs)
        matches_dist = [m.distance for m in matches]
        # print(matches_dist)

        # print('Hello 3')
        # select only good matches
        matches = [m for m in matches if m.distance < 500]

        # print(matches[5].queryIdx,matches[5].trainIdx,matches[5].distance, good_prev_kps[matches[5].queryIdx].pt, good_curr_kps[matches[5].trainIdx].pt)
        if len(matches) < 10:
            print(f'Only {len(matches)} found!!')
            continue

        # estimate motion from feature matches
        src_pts = np.array([good_prev_kps[m.queryIdx] for m in matches])
        dst_pts = np.array([good_curr_kps[m.trainIdx] for m in matches])
        # print(np.shape(src_pts), np.shape(dst_pts))
        E, mask = cv2.findEssentialMat(good_curr_pts[:,np.newaxis,:], good_prev_pts[:,np.newaxis,:], cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        if E is None:
            print(f'Essential matrix could not be computed')
            continue

        _, R, t, mask = cv2.recoverPose(E, good_curr_pts[:,np.newaxis,:], good_prev_pts[:,np.newaxis,:], K, mask=mask)

        # print(R, t)

        # P1 = np.dot(K, np.hstack((R_total, t_total)))
        # P2 = np.dot(K, np.hstack((np.dot(R_total, R), t_total + np.dot(R_total, t))))
        # pts_3d = triangulate_points(good_prev_pts[:,np.newaxis,:], good_curr_pts[:,np.newaxis,:], P1, P2)

        # if pts_3d.size > 0:
        # Compute relative scale
        # scale = np.median(pts_3d[:, 2])
        scale = 1

        # Update camera pose
        t_total += scale * np.dot(R_total, t) #[:, np.newaxis]
        R_total = np.dot(R_total, R)

        r = Rotation.from_matrix(R_total)
        angles = r.as_euler("zyx", degrees=True)

        print(f"Frame - Rotation:\n{R_total}\nTranslation:\n{t_total}\nYaw: {angles[0]} deg, Pitch: {angles[1]} deg, Roll: {angles[2]} deg\n\n")
        # else:
        #     print(f"Frame - No valid triangulated points")

        # update keypoints and descriptors for next frame
        prev_kps, prev_descs = sift.detectAndCompute(gray, None)
        prev_pts = np.array([kp.pt for kp in prev_kps], dtype=np.float32).reshape(-1, 1, 2)

        # prev_kps = good_curr_kps
        # prev_descs = curr_descs

        # save current frame and feature points
        prev_frame = gray.copy()

        # display frame
        for kp in prev_kps:        
            cv2.circle(frame, (int(kp.pt[0]), int(kp.pt[1])), 3, (0, 0, 255), -1)
    cv2.imshow('Visual Odometry', frame)

    # check for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break
    
    frame_count += 1

# release video stream and close window
cap.release()
cv2.destroyAllWindows()
