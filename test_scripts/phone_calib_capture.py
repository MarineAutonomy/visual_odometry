import cv2
import os, shutil

# set up video stream from Android phone camera over USB
# cap = cv2.VideoCapture('http://192.168.0.100:8080/video')
cap = cv2.VideoCapture('http://0.0.0.0:4747/mjpegfeed?640x480')


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

calib_flag = True

path_to_calib_folder = '/home/abhilash/Academics/github/visual_odometry/test_scripts/data/android/calib'
if calib_flag is True:
    if os.path.exists(path_to_calib_folder):
        shutil.rmtree(path_to_calib_folder)
    os.mkdir(path_to_calib_folder)


calib_count = 0

while True:
    # read frame from video stream
    ret, frame = cap.read()

    # display frame
    cv2.imshow('Android Camera Stream', frame)

    # check for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break
    
    if cv2.waitKey(1) == ord('c'):
        cv2.imwrite(f'{path_to_calib_folder}/calibration_{calib_count:02d}.jpeg', frame)
        print(f'Calibration Image {calib_count:02d} saved!')
        calib_count += 1

# release video stream and close window
cap.release()
cv2.destroyAllWindows()