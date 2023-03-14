import cv2
import depthai as dai
import numpy as np

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
    
    # Define and name output depth map
    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")

    # Define and name output disparity map
    xoutDisp = pipeline.createXLinkOut()
    xoutDisp.setStreamName("disparity")

    # Set output Xlink for left camera
    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")
 
    # Set output Xlink for right camera
    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")
  
    stereo.disparity.link(xoutDisp.input)
    stereo.depth.link(xoutDepth.input)
    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereo.rectifiedRight.link(xoutRectifiedRight.input)

    with dai.Device(pipeline) as device:
        # Get output queues. 
        disparityQueue = device.getOutputQueue(name="disparity",maxSize=1,blocking=False)
        depthQueue = device.getOutputQueue(name="depth",maxSize=1,blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft",maxSize=1,blocking=False)
        rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight",maxSize=1,blocking=False)
    
        # Calculate a multiplier for colormapping disparity map
        disparityMultiplier = 255/stereo.initialConfig.getMaxDisparity()

        # Set display window name
        cv2.namedWindow("Stereo Pair")
        cv2.setMouseCallback("Stereo Pair", mouseCallback)

        # Variable used to toggle between side by side view and one frame view. 
        sideBySide = False

        while True:

            # Get disparity map
            disparity = getFrame(disparityQueue)

            disparity = (disparity * disparityMultiplier).astype(np.uint8)
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

            # Get depth map
            depth = getFrame(depthQueue)
            # depth = (depth * depthMultiplier).astype(np.uint8)
            # depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

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
        
            imOut = cv2.cvtColor(imOut, cv2.COLOR_GRAY2RGB)
            
            imOut = cv2.line(imOut, (mouseX,mouseY), (1280,mouseY), (0,0,255), 2)
            imOut = cv2.circle(imOut, (mouseX,mouseY), 2, (255,255,128), 2)
            cv2.imshow("Stereo Pair", imOut)
            cv2.imshow("Disparity", disparity)
            cv2.imshow("Depth", depth)
            
            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                # Quit when q is pressed
                break
            elif key == ord('t'):
                # Toggle display when t is pressed
                sideBySide = not sideBySide