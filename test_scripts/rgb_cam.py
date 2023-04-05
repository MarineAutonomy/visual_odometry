#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai

def getFrame(queue):
  # Get frame from queue
  frame = queue.get()
  # Convert frame to OpenCV format and return
  return frame.getCvFrame()

def getRGBCamera(pipeline):
  # Configure RGB camera
  camRgb = pipeline.create(dai.node.ColorCamera)
 
  # Set Camera Resolution
  camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

  # set fixed focus
  # 120-130 seems appropriate for wave basin
  camRgb.initialControl.setManualFocus(130) # 0..255

  camRgb.setIspScale(1, 3)  # 1920x1080 -> 1280x720
  
  return camRgb

if __name__ == '__main__':
  # Create a pipeline
  pipeline = dai.Pipeline()

  # Create RGB camera object
  camRGB = getRGBCamera(pipeline)

  # Set output Xlink for the camera
  xout = pipeline.create(dai.node.XLinkOut)
  xout.setStreamName("RGB")

  # Attach camera to output Xlink
  camRGB.isp.link(xout.input)

  with dai.Device(pipeline) as device:
    # Read camera parameters
    calibData = device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.CENTER)

    focal_length = intrinsics[0][0]
    print('Color camera focal length in pixels:', focal_length)

    # Get output Queue
    rgbQueue = device.getOutputQueue(name="RGB", maxSize=1)

    while True:
      # Get camera frame
      rgbFrame = getFrame(rgbQueue)

      gray = cv2.cvtColor(rgbFrame, cv2.COLOR_BGR2GRAY)

      # create SIFT feature extractor
      sift = cv2.xfeatures2d.SIFT_create()

      # detect features from the image
      keypoints, descriptors = sift.detectAndCompute(rgbFrame, None)

      # draw the detected key points
      sift_image = cv2.drawKeypoints(gray, keypoints, rgbFrame)

      # show the image
      cv2.imshow('image', sift_image)
      # save the image
      # cv2.imwrite("table-sift.jpg", sift_image)
      # cv2.waitKey(0)
  
      # # Display output image
      # cv2.imshow("RGB image", rgbFrame)
      
      # Check for keyboard input
      key = cv2.waitKey(1)
      if key == ord('q'):
          # Quit when q is pressed
          break

