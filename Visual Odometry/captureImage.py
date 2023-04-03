import depthai as dai
import cv2

pipeline = dai.Pipeline()

colorCam = pipeline.create(dai.node.ColorCamera)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

xlinkOut = pipeline.create(dai.node.XLinkOut)
xlinkOut.setStreamName("rgb")

colorCam.preview.link(xlinkOut.input)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue(name="rgb",maxSize=1,blocking=False)
    image = q_rgb.get().getCvFrame()
    cv2.imwrite("image2.jpg",image)
