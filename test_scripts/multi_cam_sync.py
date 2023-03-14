#!/usr/bin/env python3

import cv2
import math
import depthai as dai
import contextlib
import argparse
import numpy as np
from datetime import timedelta

parser = argparse.ArgumentParser(epilog='Press C to capture a set of frames.')
parser.add_argument('-f', '--fps', type=float, default=30,
                    help='Camera sensor FPS, applied to all cams')

args = parser.parse_args()

cam_socket_opts = {
    'rgb'  : dai.CameraBoardSocket.RGB,
    'left' : dai.CameraBoardSocket.LEFT,
    'right': dai.CameraBoardSocket.RIGHT,
}
cam_instance = {
    'rgb'  : 0,
    'left' : 1,
    'right': 2,
}

def create_pipeline(cam_list):
    # Start defining a pipeline
    pipeline = dai.Pipeline()
    cam = {}
    xout = {}
    for c in cam_list:
        xout[c] = pipeline.create(dai.node.XLinkOut)
        xout[c].setStreamName(c)
        if c == 'rgb':
            cam[c] = pipeline.create(dai.node.ColorCamera)
            cam[c].setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam[c].initialControl.setManualFocus(130) # 0..255
            cam[c].setIspScale(1, 3)  # 1920x1080 -> 1280x720
            cam[c].isp.link(xout[c].input)
        else:
            cam[c] = pipeline.create(dai.node.MonoCamera)
            cam[c].setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            cam[c].out.link(xout[c].input)
        cam[c].setBoardSocket(cam_socket_opts[c])
        cam[c].setFps(args.fps)
    return pipeline


def mouseCallback(event,x,y,flags,param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y


# https://docs.python.org/3/library/contextlib.html#contextlib.ExitStack
with contextlib.ExitStack() as stack:
    device_infos = dai.Device.getAllAvailableDevices()

    if len(device_infos) == 0: raise RuntimeError("No devices found!")
    else: print("Found", len(device_infos), "devices")
    queues = []

    for device_info in device_infos:
        # Note: the pipeline isn't set here, as we don't know yet what device it is.
        # The extra arguments passed are required by the existing overload variants
        openvino_version = dai.OpenVINO.Version.VERSION_2021_4
        usb2_mode = False
        device = stack.enter_context(dai.Device(openvino_version, device_info, usb2_mode))

        stereo = 1 < len(device.getConnectedCameras())
        # cam_list = {'rgb', 'left', 'right'} if stereo else {'rgb'}
        cam_list = {'rgb'}

        # Get a customized pipeline based on identified device type
        device.startPipeline(create_pipeline(cam_list))

        # Read camera parameters
        calibData = device.readCalibration()
        intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CENTER)
        focal_length = intrinsics[1][1]

        # Output queue will be used to get the rgb frames from the output defined above
        for cam in cam_list:
            
            queues.append({
                'queue': device.getOutputQueue(name=cam, maxSize=4, blocking=False),
                'msgs': [], # Frame msgs
                'mx': device.getMxId(),
                'cam': cam,
                'f': focal_length
            })


    def check_sync(queues, timestamp):
        matching_frames = []
        for q in queues:
            for i, msg in enumerate(q['msgs']):
                time_diff = abs(msg.getTimestamp() - timestamp)
                # So below 17ms @ 30 FPS => frames are in sync
                if time_diff <= timedelta(milliseconds=math.ceil(500 / args.fps)):
                    matching_frames.append(i)
                    break

        if len(matching_frames) == len(queues):
            # We have all frames synced. Remove the excess ones
            for i, q in enumerate(queues):
                q['msgs'] = q['msgs'][matching_frames[i]:]
            return True
        else:
            return False
        
    # Variable used to toggle between side by side view and one frame view. 
    sideBySide = False

    mouseX = 0
    mouseY = 0

    # Set display window name
    cv2.namedWindow("Stereo Pair")
    cv2.setMouseCallback("Stereo Pair", mouseCallback)

    count = 0

    while True:
        for q in queues:
            new_msg = q['queue'].tryGet()
            if new_msg is not None:
                q['msgs'].append(new_msg)
                if check_sync(queues, new_msg.getTimestamp()):
                    leftFrame = queues[0]['msgs'].pop(0).getCvFrame()
                    left_focal = queues[0]['f']
                    rightFrame = queues[1]['msgs'].pop(0).getCvFrame()
                    right_focal = queues[1]['f']
                    print(f'right_focal : {right_focal}, left_focal : {left_focal}')
                    
                    leftgray = cv2.cvtColor(leftFrame, cv2.COLOR_BGR2GRAY)
                    rightgray = cv2.cvtColor(rightFrame, cv2.COLOR_BGR2GRAY)
                    
                    if sideBySide:
                        # Show side by side view
                        imOut = np.hstack((leftgray, rightgray))
                    else : 
                        # Show overlapping frames
                        imOut = np.uint8(leftgray/2 + rightgray/2)
                    # print(imOut[mouseY,mouseX])
                    # cv2.imwrite('/home/vallabh/visual_odometry/tutorials/data/left_image_'+str(count)+'.png',leftgray)
                    # cv2.imwrite('/home/vallabh/visual_odometry/tutorials/data/right_image_'+str(count)+'.png',rightgray)
                    count += 1
                    imOut = cv2.circle(imOut, (mouseX,mouseY), 2, (255,255,128), 2)
                    cv2.imshow("Stereo Pair", imOut)
        if cv2.waitKey(1) == ord('q'):
            break
        elif cv2.waitKey(1) == ord('t'):
                # Toggle display when t is pressed
                sideBySide = not sideBySide

