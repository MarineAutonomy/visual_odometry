import cv2
import depthai as dai
import numpy as np
from disparity import StereoSGBM
from rectify import RectifyCameras

pipeline = dai.Pipeline()

monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)

monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
leftStream = pipeline.create(dai.node.XLinkOut)
rightStream = pipeline.create(dai.node.XLinkOut)

leftStream.setStreamName("leftMono")
rightStream.setStreamName("rightMono")

monoLeft.out.link(leftStream.input)
monoRight.out.link(rightStream.input)

rectification = True

with dai.Device(pipeline) as device:

    lq = device.getOutputQueue("leftMono", blocking=False, maxSize=2)
    rq = device.getOutputQueue("rightMono", blocking=False, maxSize=2)
    calibObj = device.readCalibration()
    Ml = np.array(calibObj.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, resizeWidth=monoLeft.getResolutionWidth(),
                                               resizeHeight=monoRight.getResolutionHeight()))
    Mr = np.array(calibObj.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, resizeWidth=monoRight.getResolutionWidth(),
                                               resizeHeight=monoRight.getResolutionHeight()))
    Dl = np.array(calibObj.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
    Dr = np.array(calibObj.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
    R = np.array(calibObj.getCameraExtrinsics(dai.CameraBoardSocket.LEFT, dai.CameraBoardSocket.RIGHT))
    T = R[:3, -1]
    R = R[:3, :3]
    width = monoRight.getResolutionWidth()
    height = monoRight.getResolutionHeight()
    focalLength = Mr[0][0]
    baseline = np.abs(T[0] * 10)
    rectify = RectifyCameras(Ml, Mr, Dl, Dr, R, T, width, height)
    stereoMatcher = StereoSGBM(baseline, focalLength, H_right=rectify.giveHomographyMatrix()[1],
                               H_left=rectify.giveHomographyMatrix()[0])
    while True:
        print(device.getUsbSpeed().name)
        lframe = lq.get().getCvFrame()
        rframe = rq.get().getCvFrame()
        stereoMatcher.create_disparity_map(lframe, rframe, rectification)
        if cv2.waitKey(1) == ord('q'):
            break
