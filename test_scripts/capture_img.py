import cv2
import depthai as dai
import numpy as np

dist = "1C2" # Type Image name to be saved

def getFrame(queue):
  # Get frame from queue
  frame = queue.get()
  # Convert frame to OpenCV format and return
  return frame.getCvFrame()

def getMonoCamera(pipeline, isLeft):
  # Configure mono camera
  mono = pipeline.createMonoCamera()
 
  # Set Camera Resolution
  mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
  
  if isLeft:
      # Get left camera
      mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
  else :
      # Get right camera
      mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
  return mono

def getStereoPair(pipeline, monoLeft, monoRight):

    stereo = pipeline.createStereoDepth()
    stereo.setLeftRightCheck(True)

    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    return stereo

def mouseCallback(event,x,y,flags,param):
    global mouseX, mouseY
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y

def getRGBCamera(pipeline):
  # Configure RGB camera
  camRgb = pipeline.create(dai.node.ColorCamera)
 
  # Set Camera Resolution
  camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

  # set fixed focus
  # 120-130 seems appropriate for wave basin
  camRgb.initialControl.setManualFocus(130) # 0..255

#   camRgb.setIspScale(1, 3)  # 1920x1080 -> 1280x720
  
  return camRgb

if __name__ == '__main__':

    mouseX = 0
    mouseY = 640

    # Start defining the pipleine
    pipeline = dai.Pipeline()
 
    # Set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft = True)
    monoRight = getMonoCamera(pipeline, isLeft = False)
 
    # Combine left and right cameras to form a stero pair
    stereo = getStereoPair(pipeline, monoLeft, monoRight)

    # Create RGB camera object
    camRGB = getRGBCamera(pipeline)

    # Set output Xlink for the camera
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("RGB")

    # Attach RGB camera to output Xlink
    camRGB.isp.link(xout_rgb.input)

    # Set output Xlink for left camera
    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")
 
    # Set output Xlink for right camera
    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")
  
    # stereo.disparity.link(xoutDisp.input)
    # stereo.depth.link(xoutDepth.input)
    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereo.rectifiedRight.link(xoutRectifiedRight.input)

    with dai.Device(pipeline, usb2Mode=True) as device:
        # Get output queues. 
        # disparityQueue = device.getOutputQueue(name="disparity",maxSize=1,blocking=False)
        # depthQueue = device.getOutputQueue(name="depth",maxSize=1,blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft",maxSize=1,blocking=False)
        rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight",maxSize=1,blocking=False)
        # Get output Queue
        rgbQueue = device.getOutputQueue(name="RGB", maxSize=1)

        sideBySide = False

        while True:

            # Get camera frame
            rgbFrame = getFrame(rgbQueue)
            # Get left frame
            leftFrame = getFrame(rectifiedLeftQueue)
            # Get right frame 
            rightFrame = getFrame(rectifiedRightQueue)
        
            if sideBySide:
                # Show side by side view
                imOut = np.hstack((leftFrame, rightFrame))
            else : 
                # Show overlapping frames
                imOut = np.uint8(leftFrame/2 + rightFrame/2)
            
            # imOut = cv2.line(imOut, (mouseX,mouseY), (1280,mouseY), (0,0,255), 2)
            # imOut = cv2.circle(imOut, (mouseX,mouseY), 2, (255,255,128), 2)
            cv2.imshow("Left cam", leftFrame)
            cv2.imshow("Right cam", rightFrame)
            cv2.imshow("RBG cam", rgbFrame)

            # save the image
            cv2.imwrite(f"left {dist}.jpg", leftFrame)
            cv2.imwrite(f"right {dist}.jpg", rightFrame)
            cv2.imwrite(f"rgb {dist}.jpg", rgbFrame)
            
            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                # Quit when q is pressed
                break
            elif key == ord('t'):
                # Toggle display when t is pressed
                sideBySide = not sideBySide