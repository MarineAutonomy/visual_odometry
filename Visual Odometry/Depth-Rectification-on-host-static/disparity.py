import cv2
import numpy as np
import depthai as dai


class StereoSGBM:
    def __init__(self, baseline, focalLength, H_right, H_left=np.identity(3, dtype=np.float32)):
        self.disparity = None
        self.max_disparity = 192
        self.blockSize = 11
        self.stereoProcessor = cv2.StereoSGBM_create(
            minDisparity=1,
            numDisparities=self.max_disparity,
            blockSize=11,
            # speckleWindowSize=200,
            # speckleRange=2,
            P1=8 * self.blockSize * self.blockSize,  # 50
            P2=32 * self.blockSize * self.blockSize,  # 800
            disp12MaxDiff=5,
            mode=cv2.STEREO_SGBM_MODE_HH4
        )
        self.baseline = baseline
        self.focal_length = focalLength
        self.H1 = H_left  # for left camera
        self.H2 = H_right  # for right camera

    def rectification(self, left_img, right_img):
        # warp right image
        img_l = cv2.warpPerspective(left_img, self.H1, left_img.shape[::-1],
                                    cv2.INTER_CUBIC +
                                    cv2.WARP_FILL_OUTLIERS +
                                    cv2.WARP_INVERSE_MAP)

        img_r = cv2.warpPerspective(right_img, self.H2, right_img.shape[::-1],
                                    cv2.INTER_CUBIC +
                                    cv2.WARP_FILL_OUTLIERS +
                                    cv2.WARP_INVERSE_MAP)
        return img_l, img_r

    def create_disparity_map(self, left_img, right_img, is_rectify_enabled=True):

        if is_rectify_enabled:
            left_img_rect, right_img_rect = self.rectification(left_img, right_img)  # Rectification using Homography
        else:
            left_img_rect = left_img
            right_img_rect = right_img

        # opencv skips disparity calculation for the first max_disparity pixels
        padImg = np.zeros(shape=[left_img.shape[0], self.max_disparity], dtype=np.uint8)
        left_img_rect_pad = cv2.hconcat([padImg, left_img_rect])
        right_img_rect_pad = cv2.hconcat([padImg, right_img_rect])
        self.disparity = self.stereoProcessor.compute(left_img_rect_pad, right_img_rect_pad)
        self.disparity = self.disparity[0:self.disparity.shape[0], self.max_disparity:self.disparity.shape[1]]

        # scale back to integer disparities, opencv has 4 subpixel bits
        print(f"lmao {np.max(self.disparity)}")
        disparity_scaled = (self.disparity / 16.).astype(np.uint8)
        return disparity_scaled
        # disparity_colour_mapped = cv2.applyColorMap(
        #     (disparity_scaled * (256. / self.max_disparity)).astype(np.uint8),
        #     cv2.COLORMAP_HOT)
        # cv2.imshow("Disparity", disparity_colour_mapped)
        # left_img_rect = cv2.cvtColor(left_img_rect,cv2.COLOR_GRAY2BGRA)
        # right_img_rect = cv2.cvtColor(right_img_rect,cv2.COLOR_GRAY2BGRA)
        # height, width = np.shape(left_img)
        # for i in range(8):
        #     left_img_rect = cv2.line(left_img_rect, (0, 50 * (i + 1)), (width, 50 * (i + 1)), color=(0, 255, 0),thickness=2)
        # for i in range(8):
        #     right_img_rect = cv2.line(right_img_rect, (0, 50 * (i + 1)), (width, 50 * (i + 1)), color=(0, 255, 0),thickness=2)
        # if is_rectify_enabled:
        #     cv2.imshow("rectified left", left_img_rect)
        #     cv2.imshow("rectified right", right_img_rect)
        # else:
        #     cv2.imshow("not-rectified left", left_img_rect)
        #     cv2.imshow("not-rectified right", right_img_rect)
