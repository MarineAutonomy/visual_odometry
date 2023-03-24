import cv2
import depthai as dai
import numpy as np
from scipy.spatial.transform import Rotation
import sys, os


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

in_rgb = q_rgb.tryGet()
frame = in_rgb.getCvFrame()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
prev_frame = gray
# prev_keypoints, prev_descriptors = orb.detectAndCompute(prev_frame, None)
prev_keypoints, prev_descriptors = sift.detectAndCompute(prev_frame, None)

frame_count = 0
frame_flag = 0
while True:
    # Get the image from the rgb queue
    in_rgb = q_rgb.tryGet()

    if in_rgb is not None:
        # Convert image to grayscale
        frame = in_rgb.getCvFrame()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)       

        # Visualize feature matches        
        cv2.imshow("Visual Odometry", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

    frame_count += 1
    frame_flag = 0

# Cleanup
cv2.destroyAllWindows()
