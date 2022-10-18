import cv2
import numpy as np
import cv2.aruco as aruco
import time
from matplotlib import pyplot as plt

chessboard_size = (6, 4)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def draw_frames(frame_left, frame_right):
    numpy_horizontal_concat = np.concatenate((frame_left, frame_right), axis=1)
    cv2.imshow("0", numpy_horizontal_concat)
    cv2.waitKey(1)

def find_corners_charuco_two_cams(video_device_left, video_device_right, detection_length):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    run = "./runs/run1/"

    object_points = np.zeros((np.prod(chessboard_size), 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

    counter, corners_list_left, corners_list_right, objpts = [], [], [], []
    frames_left, frames_right = [], []

    detection_counter = 0
    clear_buffer = 0
    found_left = found_right = False
    while detection_counter < detection_length:
        _, frame_left = video_device_left.read()
        _, frame_right = video_device_right.read()

        if clear_buffer > 0:
            clear_buffer -= 1
            time.sleep(0.5)
            continue

        frame_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        frame_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        draw_frames(frame_left, frame_right)

        ### camera left
        found, corners_left = cv2.findChessboardCorners(frame_left, chessboard_size, None)
        if found and len(corners_left) == 24:
            cv2.cornerSubPix(frame_left, corners_left, (11, 11), (-1, -1), criteria)
            assert found
            found_left = True
            print("Found left")

        ### camera right
        found, corners_right = cv2.findChessboardCorners(frame_right, chessboard_size, None)
        if found and len(corners_right) == 24:
            cv2.cornerSubPix(frame_right, corners_right, (11, 11), (-1, -1), criteria)
            assert found
            found_right = True
            print("Found right")

        if found_left and found_right:
            cv2.imwrite(run + "myleft{}.png".format(detection_counter), frame_left)
            cv2.imwrite(run + "myright{}.png".format(detection_counter), frame_right)
            detection_counter += 1
            print("Board found ({}/{})".format(detection_counter, detection_length))
            frames_left.append(frame_left)
            corners_list_left.append(corners_left.reshape(-1, 2))
            frames_right.append(frame_right)
            corners_list_right.append(corners_right.reshape(-1, 2))

            assert len(corners_list_left) == len(corners_list_right) == detection_counter
            objpts.append(object_points)
            clear_buffer = 10
        found_left = found_right = False

    return corners_list_left, corners_list_right, frame_right.shape, objpts, frames_left, frames_right

def draw_epilines(pts1, pts2, F, img1, img2):
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()


def calibrate(image_points_left, image_points_right, image_size, object_points):
    ## Calibrate cameras
    (cam_mats, dist_coefs, rect_trans, proj_mats, valid_boxes,
     undistortion_maps, rectification_maps) = {}, {}, {}, {}, {}, {}, {}
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS,
                100, 1e-5)
    flags = (cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_ZERO_TANGENT_DIST +
             cv2.CALIB_SAME_FOCAL_LENGTH)
    (ret, cam_mats["left"], dist_coefs["left"], cam_mats["right"],
     dist_coefs["right"], rot_mat, trans_vec, e_mat,
     f_mat) = cv2.stereoCalibrate(object_points,
                                  image_points_left, image_points_right, None, None, None, None,
                                  image_size, criteria=criteria, flags=flags)

    print('error: {}'.format(ret))
    return f_mat, cam_mats, dist_coefs



if __name__ == "__main__":
    video_capture_left = cv2.VideoCapture(6)
    video_capture_right = cv2.VideoCapture(2)
    corners_list_left, corners_list_right, image_size, objpts, frames_left, frames_right = find_corners_charuco_two_cams(video_capture_left,
                                                                                          video_capture_right,  4)

    f_mat, cam_mats, dist_coefs = calibrate(corners_list_left, corners_list_right, image_size, objpts)

   # corners_list_left = cv2.undistortPoints(np.array(corners_list_left), cam_mats["left"], dist_coefs["left"], None, cam_mats["left"])

    np.save("./camera_params/stereo/F", f_mat)
    draw_epilines(np.int32(corners_list_left[0]), np.int32(corners_list_right[0]), f_mat, frames_left[0], frames_right[0])

    draw_epilines(np.int32(corners_list_left[1]), np.int32(corners_list_right[1]), f_mat, frames_left[1], frames_right[1])

    video_capture_left.release()
    video_capture_right.release()
