#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
# import rospy
import time
from scipy.spatial.transform import Rotation   
import matplotlib.pyplot as plt

class VisualOdometryNode():

    def __init__(self, \
                ros_dependent=True, \
                manual_focus=None, \
                fps=None, detector=cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)\
                    ):
        
        self.detector = detector        
        self.pipeline = self.create_pipeline_single_rgb_camera()
        self.ros_dependent = ros_dependent        
        self.intrinsics = None        
        self.prev_frame = None
        self.curr_frame = None
        self.lk_params = dict(winSize  = (21,21), criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.n_features = 0
        self.R = np.eye(3, dtype=float)
        self.t = np.zeros(shape=(3,1), dtype=float)
        self.good_prev = None
        self.good_curr = None
        self.P = np.zeros((3,4))
        self.estimated_path = []
        self.cur_pose = np.eye(4, dtype=float)
        self.img3 = None
        self.currentR = self.R
         
    def detect(self, img):
        # Used to detect features and parse into useable format
        #
        # Arguments:
        # img {np.ndarray} -- Image for which to detect keypoints on
        #
        # Returns:
        # np.array -- A sequence of points in (x, y) coordinate format
        # denoting location of detected keypoint
        
        p0 = self.detector.detect(img)

        return np.array([x.pt for x in p0], dtype=np.float32).reshape(-1, 1, 2)

    def _form_transf(self, R, t):
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
        T[:3, 3] = t
        return T
    
    def sum_z_cal_relative_scale(self, R, t, q1, q2):
        # Get the transformation matrix
        T = self._form_transf(R, t)
        # Make the projection matrix
        P = np.matmul(np.concatenate((self.intrinsics, np.zeros((3, 1))), axis=1), T)

        # Triangulate the 3D points
        hom_Q1 = cv2.triangulatePoints(self.P, P, q1.T, q2.T)

        # print(hom_Q1)
        # Also seen from cam 2
        hom_Q2 = np.matmul(T, hom_Q1)

        # Un-homogenize
        uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        # Find the number of points there has positive z coordinate in both cameras
        sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
        sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

        # Form point pairs and calculate the relative scale
        relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/
                                    np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
        
        # print(relative_scale)
        
        self.P = P
        return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale
    
    def decomp_essential_mat(self, E, q1, q2, recover_pose):
        """
        Decompose the Essential matrix
        Parameters
        ----------
        E (ndarray): Essential matrix
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        Returns
        -------
        right_pair (list): Contains the rotation matrix and translation vector
        """


        if recover_pose:
            _, R, t, _ = cv2.recoverPose(E, self.good_prev, self.good_curr, self.intrinsics)
            t = np.squeeze(t)
            z_sum, scale = self.sum_z_cal_relative_scale(R, t, q1, q2)
            t = t * scale
            return [R,t]
        else:
            # Decompose the essential matrix
            R1, R2, t = cv2.decomposeEssentialMat(E)
            t = np.squeeze(t)

            # Make a list of the different possible pairs
            pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

            # Check which solution there is the right one
            z_sums = []
            relative_scales = []
            for R, t in pairs:
                z_sum, scale = self.sum_z_cal_relative_scale(R, t, q1, q2)
                z_sums.append(z_sum)
                relative_scales.append(scale)

            # print(z_sums)
            
            # Select the pair there has the most points with positive z coordinate
            right_pair_idx = np.argmax(z_sums)
            right_pair = pairs[right_pair_idx]
            relative_scale = relative_scales[right_pair_idx]

            # print(relative_scale)qq

            R1, t = right_pair
            t = t * relative_scale

            return [R1, t]

    def calc_vo(self):
        if self.n_features < 2000:
            self.p0 = self.detect(self.prev_frame)
            self.n_features = np.shape(self.p0)[0]

        # Calculate optical flow between frames, st holds status
        # of points from frame to frame
        # st = 1 for matches
        self.p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_frame, self.curr_frame, self.p0, None, **self.lk_params)        
        
        # Save the good points from the optical flow
        self.good_prev = self.p0[st == 1]
        self.good_curr = self.p1[st == 1]

        for k in range(len(self.good_prev)):
            self.img3 = cv2.circle(self.prev_frame, (self.good_prev[k,0], self.good_prev[k,1]), 2, (255,255,128), 2)

        # If the frame is one of first two, we need to initalize
        # our t and R vectors so behavior is different
        E, _ = cv2.findEssentialMat(self.good_prev, self.good_curr, self.intrinsics, cv2.RANSAC, 0.999, 1.0)

        # Decompose the Essential matrix into R and t
        self.R, self.t = self.decomp_essential_mat(E, self.good_prev, self.good_curr, recover_pose = True)
        # print("t: ", self.t)

        # Get transformation matrix
        transformation_matrix = self._form_transf(self.R, np.squeeze(self.t))

        # print(transformation_matrix)
        
        self.cur_pose = np.matmul(self.cur_pose, np.linalg.inv(transformation_matrix))

        self.currentR = self.cur_pose[:3,:3]
        r =  Rotation.from_matrix(self.currentR)
        angles = r.as_euler("zyx",degrees=True)
        print("Rotation angles", angles)

        self.estimated_path.append((self.cur_pose[0,3], self.cur_pose[1,3], self.cur_pose[2,3]))
        
        # Save the total number of good features
        self.n_features = self.good_curr.shape[0]

    def execute(self):

        with dai.Device(self.pipeline) as device:
            # Read camera parameters
            calib = device.readCalibration()
            self.intrinsics = np.array(calib.getCameraIntrinsics(dai.CameraBoardSocket.CENTER))

            focal_length = self.intrinsics[0][0]
            print('INFO : Color camera focal length in pixels:', focal_length)

            # Get output Queue
            rgb_queue = device.getOutputQueue(name="oak_rgb", maxSize=1)

            # if self.ros_dependent is False:
            #     flag = True
            # else:
            #     flag = not rospy.is_shutdown()
            flag = True

            # time.sleep(10)
            counter = 0
            gap_count = 0
            while flag:
                # Get camera frame
                rgb_frame = rgb_queue.get().getCvFrame()

                # Get current frame in grayscale
                self.curr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

                # To be executed only once at beginning as prev_frame is not set at beginning
                if self.prev_frame is None:
                    self.prev_frame = self.curr_frame

                if counter > 100:
                    try:
                        if gap_count > 1:
                            self.calc_vo()
                            cv2.imshow('image', self.img3)
                            self.prev_frame = self.curr_frame
                            gap_count = 0
                    except:
                        pass
                else:
                    self.prev_frame = self.curr_frame
                
                # Check for keyboard input
                key = cv2.waitKey(1)
                if key == ord('q'):
                    # Quit when q is pressed
                    break


                gap_count += 1
                counter += 1


    def create_pipeline_single_rgb_camera(self, manual_focus=None, fps=None):
        # Create a pipeline
        pipeline = dai.Pipeline()

        # Configure RGB camera object
        cam_rgb = pipeline.create(dai.node.ColorCamera)

        # Set Camera Resolution
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

        # set fixed focus
        # 120-130 seems appropriate for wave basin
        if manual_focus is not None:
            cam_rgb.initialControl.setManualFocus(manual_focus) # 0..255

        cam_rgb.setIspScale(1, 3)  # 1920x1080 -> 1280x720

        # Set output Xlink for the camera
        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("oak_rgb")

        # Attach camera to output Xlink
        cam_rgb.isp.link(xout.input)

        if fps is not None:
            cam_rgb.setFps(fps)
        else:
            cam_rgb.setFps(30)

        return pipeline


if __name__=="__main__":
    vo_node = VisualOdometryNode(ros_dependent=False, fps=0.1)
    
    vo_node.execute()

    path = np.array(vo_node.estimated_path)
    print(path)

    x = path[:,0]
    y = path[:,1]
    z = path[:,2]

    print(np.shape(x))
    print(np.shape(y))
    print(np.shape(z))

    plt.plot(x, y)
    # plt.plot(x, z)
    # plt.plot(y, z)
    plt.show()
