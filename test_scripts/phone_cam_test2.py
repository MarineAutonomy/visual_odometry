import cv2

# set up video stream from Android phone camera over USB
cap = cv2.VideoCapture(0)

# count = 1
# while True:
    
#     ret, frame = cap.read()
#     if frame is None:
#         print(f'No camera feed found in device {count}')
#         cap.release()
#         count += 1
#     else:
#         print(f'Camera feed found in device {count}')
#         break

# loop over frames from video stream
while True:
    # read frame from video stream
    ret, frame = cap.read()

    # display frame
    cv2.imshow('Android Camera Stream', frame)

    # check for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break

# release video stream and close window
cap.release()
cv2.destroyAllWindows()