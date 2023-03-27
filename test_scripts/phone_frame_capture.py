import cv2
import os, shutil

# set up video stream from Android phone camera over USB
cap = cv2.VideoCapture('http://0.0.0.0:4747/mjpegfeed?640x480')

path_to_data_folder = '/home/abhilash/Academics/github/visual_odometry/test_scripts/data/android/data'

if not os.path.exists(path_to_data_folder):
    os.mkdir(path_to_data_folder)

data_count = 0

while True:
    # read frame from video stream
    ret, frame = cap.read()

    # display frame
    cv2.imshow('Android Camera Stream', frame)

    # check for key press to exit
    if cv2.waitKey(1) == ord('q'):
        break
    
    if cv2.waitKey(1) == ord('c'):
        cv2.imwrite(f'{path_to_data_folder}/data_{data_count:02d}.jpeg', frame)
        print(f'Data Image {data_count:02d} saved!')
        data_count += 1

# release video stream and close window
cap.release()
cv2.destroyAllWindows()