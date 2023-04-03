import cv2
import numpy as np


class RectifyCameras:
    def __init__(self, M_left, M_right, D_left, D_right, R_extrinsic, T_extrinsic, width, height):
        self.H_right = None
        self.H_left = None
        self.M_right = M_right
        self.M_left = M_left
        self.D_left = D_left
        self.D_right = D_right
        self.R_extrinsic = R_extrinsic
        self.T_extrinsic = T_extrinsic
        self.width = width
        self.height = height
        self.Rl, self.Rr, _, _, _, _, _ = cv2.stereoRectify(cameraMatrix1=M_left, cameraMatrix2=M_right,
                                                            distCoeffs1=D_left,
                                                            distCoeffs2=D_right,
                                                            R=R_extrinsic,
                                                            T=T_extrinsic, imageSize=(width, height))

    def giveHomographyMatrix(self):
        self.H_left = np.matmul(np.matmul(self.M_right, self.Rl), np.linalg.inv(self.M_left))
        self.H_right = np.matmul(np.matmul(self.M_right, self.Rr), np.linalg.inv(self.M_right))
        return [self.H_left, self.H_right]
