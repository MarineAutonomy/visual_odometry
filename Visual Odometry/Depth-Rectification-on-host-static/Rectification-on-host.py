import cv2
import numpy as np
from disparity import StereoSGBM
from rectify import RectifyCameras
from justpfm import justpfm

rectification = True
pfm = justpfm.read_pfm("L_00001.pfm")
lq = cv2.imread("L_00001.png",0)
rq = cv2.imread("R_00001.png",0)

Ml = np.array([[615., 0., 320.],[0., 615., 240.],[0., 0., 1.]])
Mr =  np.array([[615., 0., 320.],[0., 615., 240.],[0., 0., 1.]])
R = np.array([[1.,0.,0.,10.],[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,0.,0.,1.]])
T = R[:3, -1]
R = R[:3, :3]
width = 640
height = 480
Dl = Dr = np.array([1.,1.,1.,1.,1.])
focalLength = Mr[0][0]
baseline = np.abs(T[0] * 10)
rectify = RectifyCameras(Ml, Mr, Dl, Dr, R, T, width, height)
stereoMatcher = StereoSGBM(baseline, focalLength, H_right=rectify.giveHomographyMatrix()[1],
                       H_left=rectify.giveHomographyMatrix()[0])
disp_levels = 1
max_disparity = 192
dispScaleFactor = baseline * focalLength * disp_levels
lframe = lq
rframe = rq
# 270 192
# 330 214
dispFrame = stereoMatcher.create_disparity_map(lframe, rframe, rectification)
calcedDepth = (dispScaleFactor / dispFrame).astype(np.uint16)
disparity_colour_mapped = cv2.applyColorMap(
    (dispFrame * (256. / max_disparity)).astype(np.uint8),
    cv2.COLORMAP_HOT)
cv2.rectangle(disparity_colour_mapped, [240, 280], [270, 310], thickness=2, color=(255, 255, 255))
print(f"Distance = {np.mean(calcedDepth[280:310, 240:270]) / 10}cm ")
print(f"Ground truth = {np.mean(pfm[280:310, 240:270])}cm")
cv2.imshow("disp", disparity_colour_mapped)
cv2.waitKey()
cv2.destroyAllWindows()
