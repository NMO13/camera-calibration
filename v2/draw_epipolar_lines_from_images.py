import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

criteria =(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

def get_files(path):
    f = {}
    for file in os.listdir(path):
        f[file] = file
    return f

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines
        code from OpenCV documentation
        '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),2,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),2,color,-1)
    return img1,img2

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

    return f_mat, e_mat, cam_mats, dist_coefs, rot_mat, trans_vec
def draw_epilines(pts1, pts2, F, img1, img2):
    ''''
    code from OpenCV documentation
    '''
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

def draw_epiline(pt, F, img0, img1):
    color = (150, 102, 187)
    line = cv2.computeCorrespondEpilines(pt.reshape(1, 1, 2), 1, F).reshape(3, )
    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    img0 = cv2.circle(img0,tuple(pt),2,color,-1)

    x0, y0 = map(int, [0, -line[2] / line[1]])
    x1, y1 = map(int, [img0.shape[1], -(line[2] + line[0] * img0.shape[1]) / line[1]])
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)

    plt.subplot(121),plt.imshow(img0)
    plt.subplot(122),plt.imshow(img1)
    plt.show()

def save_to_file(F, E, dist_left, dist_right, cam_mats_left, cam_mats_right, optimal_left, optimal_right, R, t) :
    np.save("./mycode/F", F)
    np.save("./mycode/E", E)
    np.save("./mycode/distL", dist_left)
    np.save("./mycode/distR", dist_right)
    np.save("./mycode/camLeft", cam_mats_left)
    np.save("./mycode/camRight", cam_mats_right)
    np.save("./mycode/optimLeft", optimal_left)
    np.save("./mycode/optimRight", optimal_right)
    np.save("./mycode/R", R)
    np.save("./mycode/t", t)


if __name__ == "__main__":
    path = "./images/"
    f = get_files(path)

    corners_list_left, corners_list_right, objpoints = [], [], []
    images_left, images_right = [], []

    for counter in range(int(len(f)/2)):
        try:
            grayL = cv2.imread(path + f["chessboard-L{}.png".format(counter)], 0)
            grayR = cv2.imread(path + f["chessboard-R{}.png".format(counter)], 0)
        except:
            continue

        retR, cornersR = cv2.findChessboardCorners(grayR, (9, 6), None)
        retL, cornersL = cv2.findChessboardCorners(grayL, (9, 6), None)
        assert retR == True and retL == True

        images_left.append(grayL)
        images_right.append(grayR)
        objpoints.append(objp)
        corners2R= cv2.cornerSubPix(grayR,cornersR,(11,11),(-1,-1),criteria)    # Refining the Position
        corners2L= cv2.cornerSubPix(grayL,cornersL,(11,11),(-1,-1),criteria)


        corners_list_right.append(corners2R.reshape(-1, 2))
        corners_list_left.append(corners2L.reshape(-1, 2))


        # cv2.drawChessboardCorners(grayR,(9,6),corners2R,retR)
        # cv2.drawChessboardCorners(grayL,(9,6),corners2L,retL)
        # cv2.imshow('VideoR',grayR)
        # cv2.imshow('VideoL',grayL)
        # cv2.waitKey(2000)

    F, E, cam_mats, dist_coefs, R, t = calibrate(corners_list_left, corners_list_right, grayL.shape, objpoints)

    F1, Fmask = cv2.findFundamentalMat(np.array(corners_list_left).reshape(len(corners_list_left)*54, 2),
                                                np.array(corners_list_right).reshape(len(corners_list_left)*54, 2),
                                                cv2.FM_RANSAC, 0.1, 0.99)

    hR, wR = grayR.shape[:2]
    OmtxR, roiR = cv2.getOptimalNewCameraMatrix(cam_mats["right"], dist_coefs["right"],(wR, hR), 1, (wR, hR))

    hL, wL = grayL.shape[:2]
    OmtxL, roiL = cv2.getOptimalNewCameraMatrix(cam_mats["left"], dist_coefs["left"], (wL, hL), 1, (wL, hL))

    img_undist_left = cv2.undistort(images_left[12], cam_mats["left"], dist_coefs["left"], None, OmtxL)
    img_undist_right = cv2.undistort(images_right[12], cam_mats["right"], dist_coefs["right"], None, OmtxR)

    # save_to_file(F, E, dist_coefs["left"], dist_coefs["right"], cam_mats["left"], cam_mats["right"], OmtxL, OmtxR, R, t)

    draw_epiline(np.int32(np.array([76, 398])), F, img_undist_left, img_undist_right)

  #  [draw_epilines(np.int32(corners_list_left[i][:40:2]), np.int32(corners_list_right[i][:40:2]), F, cv2.undistort(images_left[i], cam_mats["left"], dist_coefs["left"], None, OmtxL), cv2.undistort(images_right[i], cam_mats["right"], dist_coefs["right"], None, OmtxR)) for i in range(len(images_left))]