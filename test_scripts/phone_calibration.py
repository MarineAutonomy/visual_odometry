import numpy as np
import cv2 as cv
import glob

def calibrate(show_calib_images=False):

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    path_to_calib_folder = '/home/abhilash/Academics/github/visual_odometry/test_scripts/data/android/calib'
    images = glob.glob(f'{path_to_calib_folder}/*.jpeg')

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7,7), None)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            if show_calib_images is True:
                # Draw and display the corners
                cv.drawChessboardCorners(img, (7,7), corners2, ret)
                cv.imshow('Calibration Image', img)
                cv.waitKey(2000)
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    img = cv.imread(f'{path_to_calib_folder}/calibration_00.jpeg')
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite(f'{path_to_calib_folder}/calibresult.png', dst)


    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error
    reproj_err = mean_error/len(objpoints)

    return mtx, newcameramtx, reproj_err, dist, rvecs, tvecs

if __name__=="__main__":
    K_old, K_new, reproj_err, _, _, _s = calibrate()

    print('Calibration Matrix prior to distortion correction')
    print(K_old)

    print('Calibration Matrix after distortion correction')
    print(K_new)

    print( "Total reprojection error: {}".format(reproj_err) )