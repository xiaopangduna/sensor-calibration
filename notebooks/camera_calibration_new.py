import numpy as np
import cv2
import glob
import yaml
import os
import tarfile
import shutil
import random
import argparse

parser = argparse.ArgumentParser(description = 'Input Camera Calibration Settings')
parser.add_argument('--size', type = str, help = 'chessboard size', default = '8x6')
parser.add_argument('--cam_name', type = str, help = 'camera number', default = '1')
parser.add_argument('--mode', type = str, help = 'choose camera model: pinhole; fisheye', default = "fisheye")
parser.add_argument('--fisheye_mode', type = str, help = 'choose fisheye camera model: KB; Mei', default = "KB")
parser.add_argument('--result_path', type = str, help = 'save result path; format: ./*****',
                    default = "/home/ubuntu/Desktop/project/sensor-calibration/tmp/rosbag/result")
parser.add_argument('--images_path', type = str, help = 'save images path; format: ./*****',
                    default = "/home/ubuntu/Desktop/project/sensor-calibration/tmp/rosbag/images")
parser.add_argument('--show_result', type = int, help = 'visualize the calibration result, press ESC to exit, '
                                                        '"0": false "1": true', default = 1)
parser.add_argument('--test_mode', type = int, help = 'test mode 1: test mode activate', default = 1)
args = parser.parse_args()

chessboard_w, chessboard_h = [int(c) for c in args.size.split('x')]
CAMERA_NAME = args.cam_name
if args.test_mode:
    test = True
else:
    test = False
if args.show_result:
    visualize_result = True
else:
    visualize_result = False
if args.mode == "pinhole":
    fisheye = False
elif args.mode == "fisheye":
    fisheye = True

# image
IMAGE_W = 1920
IMAGE_H = 1080
# termination criteria
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-6)


def extra_data(extra_path):

    if not test:
        # remove old data
        tmp_file = "./calibrationdata.tar.gz"
        tmp_folder = "./calibrationdata"
        if os.path.exists(tmp_file) :
            os.remove(tmp_file)
            print("old images zip data deleted!")
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
            print("old images folder deleted!")

        # copy and unzip new data
        try:
            ws_dir = os.getcwd()
            shutil.copyfile('/tmp/calibrationdata.tar.gz', ws_dir + '/calibrationdata.tar.gz')
        except IOError:
            print('Error: Can`t find the calibration data file /tmp/calibrationdata.tar.gz')
            if os.path.exists('./calibrationdata.tar.gz'):
                print('Find the calibration data in current folder, continue!')
            else:
                print('Error: Can`t find the calibration data file in current folder, process end!')
                quit()
        else:
            print('calibration data copied!')

        tar_files = tarfile.open("./calibrationdata.tar.gz")
        tar_files.extractall(extra_path)
        tar_files.close()

    try:
        os.path.exists('./sample.yaml')
    except IOError:
        print('No calibration sample.yaml file')
    else:
        # if fisheye:
        #     shutil.copyfile('./sample.yaml', './ud_camera{}.yaml'.format(CAMERA_NAME))
        #     shutil.copyfile('./sample.yaml', './ud_camera{}_mei.yaml'.format(CAMERA_NAME))
        # else:
            shutil.copyfile('./sample.yaml', './ud_camera{}.yaml'.format(CAMERA_NAME))


def load_images(images):
    if not fisheye:
        objp = np.zeros((chessboard_w * chessboard_h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_w, 0:chessboard_h].T.reshape(-1, 2)
    else:
        objp = np.zeros((1, chessboard_w * chessboard_h, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:chessboard_w, 0:chessboard_h].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    print('-------------------------loading data----------------------------')
    load_count = 0
    for filename in images:
        img = cv2.imread(filename, 0)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.equalizeHist(img)
        smoothed_gray = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, corners = cv2.findChessboardCornersSB(smoothed_gray, (chessboard_w, chessboard_h),
                                                   flags = cv2.CALIB_CB_NORMALIZE_IMAGE
                                                           + cv2.CALIB_CB_EXHAUSTIVE
                                                           + cv2.CALIB_CB_LARGER
                                                   )
        # ret, corners = cv2.findChessboardCornersSBWithMeta(smoothed_gray,
        #                                                    patternSize = (chessboard_w, chessboard_h),
        #                                                    flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        #                                                            + cv2.CALIB_CB_EXHAUSTIVE
        #                                                            + cv2.CALIB_CB_LARGER,
        #                                                    meta = 1
        #                                                    )
        if ret:
            load_count += 1
            corners_subpix = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            if len(corners_subpix) == chessboard_w*chessboard_h:
                objpoints.append(objp)
                imgpoints.append(corners_subpix)
                print("loading image: " + filename)

    print("Total " + str(load_count) + "/" + str(len(images)) + " images loaded!")

    return objpoints, imgpoints


def calc_reproj_error(objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs, xi = None, idx = None):
    # with open('./calibrationdata/ost.yaml') as f:
    #     data = yaml.load(f, Loader = yaml.FullLoader)
    #     image_width = data['image_width']
    #     image_height = data['image_height']
    #     cam_matrix_ros = np.asarray(data['camera_matrix']["data"]).reshape((3, 3))
    #     dist_ros = np.asarray([data["distortion_coefficients"]["data"]])
    #     f.close()

    print("----------------------camera calibration results-----------------------")
    print("camera matrix内参:\n", mtx, "\ndistortion:\n", dist)

    total_error_MSE = 0
    total_error_RMS = 0

    if not fisheye:
        for i in range(len(objpoints)):
            imgpoints_cv, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints_cv, cv2.NORM_L2)
        total_error_MSE += error / len(imgpoints_cv)
        total_error_RMS += error * error

    elif fisheye and args.fisheye_mode == "KB":
        for i in range(len(objpoints)):
            imgpoints_cv, _ = cv2.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist, imgpoints[i])
            error = cv2.norm(imgpoints[i], imgpoints_cv.transpose(1, 0, 2), cv2.NORM_L2)
        total_error_MSE += error / len(imgpoints_cv)
        total_error_RMS += error * error

    elif fisheye and args.fisheye_mode == "Mei":
        for i in range(len(idx)):
            j = idx[0][i]
            imgpoints_cv, _ = cv2.omnidir.projectPoints(objpoints[j], rvecs[i], tvecs[i], mtx, xi[0][0], dist,
                                                        imgpoints[j])
            error = cv2.norm(imgpoints[i], imgpoints_cv.transpose(1, 0, 2), cv2.NORM_L2)

        total_error_MSE += error / len(imgpoints_cv)
        total_error_RMS += error * error

    MSE_error = total_error_MSE / len(objpoints)
    RMS_error = np.sqrt(total_error_RMS / (len(objpoints) * len(objpoints[0])))

    print("----------------------Re-projection Error-----------------------------")
    print("Note: The RMS(Root Mean Square) Re-projection Error should lower than 0.5")
    print("RMS Re-projection Error: ", RMS_error)


def save_results(save_result_path, save_images_path, mtx, dist, xi = None):
    # with open("/home/ubuntu/Desktop/project/sensor-calibration/tmp/rosbag/result/ud_camera{}.yaml".format(CAMERA_NAME), "r") as f:
    #     data_dict = yaml.load(f, Loader = yaml.FullLoader)

    data_dict = {}

    with open("/home/ubuntu/Desktop/project/sensor-calibration/tmp/rosbag/result/ud_camera{}.yaml".format(CAMERA_NAME), "w+") as f:
        camera_matrix = mtx.reshape(-1).tolist()
        distortion = dist.reshape(-1).tolist()
        data_dict['camera_matrix'] = camera_matrix
        data_dict['dist_coeffs'] = distortion
        if args.fisheye_mode == "Mei":
            xi_ = xi.reshape(-1).tolist()
            data_dict['xi'] = xi_
        yaml.dump(data_dict, f, default_flow_style = None)

    # save results and backup images data
    if not os.path.exists(save_result_path):
        os.mkdir(save_result_path)
    if fisheye and args.fisheye_mode == "Mei":
        os.rename('/home/ubuntu/Desktop/project/sensor-calibration/tmp/rosbag/result/ud_camera{}.yaml'.format(CAMERA_NAME),
                  save_result_path + '/ud_camera{}_Mei.yaml'.format(CAMERA_NAME))
    else:
        os.rename('/home/ubuntu/Desktop/project/sensor-calibration/tmp/rosbag/result/ud_camera{}.yaml'.format(CAMERA_NAME),
                  save_result_path + '/ud_camera{}.yaml'.format(CAMERA_NAME))

    if not os.path.exists(save_images_path):
        os.mkdir(save_images_path)
    # if not os.path.exists(save_images_path + '/calibrationdata_camera{}.tar.gz'.format(CAMERA_NAME)) and not test:
    #     os.rename('./calibrationdata.tar.gz',
    #               save_images_path + '/calibrationdata_camera_{}.tar.gz'.format(CAMERA_NAME))


def visualize_result(images, mtx, dist, xi = None):
    sample_idx = random.randint(0, len(images) - 1)
    sample_test = cv2.imread(images[sample_idx])
    # h, w = sample_test.shape[:2]
    if not fisheye:
        '''优化相机内参（camera matrix），这一步可选。参数1表示保留所有像素点，同时可能引入黑色像素，设为0表示尽可能裁剪不想要的像素，这是个scale，0-1都可以取。'''
        optim_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (IMAGE_W, IMAGE_H), 0,
                                                          (IMAGE_W, IMAGE_H))
        # undistort
        dst = cv2.undistort(sample_test, mtx, dist, None, optim_matrix)
    elif fisheye:
        if args.fisheye_mode == "KB":
            # optim_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, (IMAGE_W, IMAGE_H), np.eye(3),
            #                                                                       None)
            # optim_matrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (IMAGE_W, IMAGE_H), 1, (IMAGE_W, IMAGE_H), 0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, (IMAGE_W, IMAGE_H),
                                                             cv2.CV_32FC1)
        elif args.fisheye_mode == "Mei":

            map1, map2 = cv2.omnidir.initUndistortRectifyMap(mtx, dist, xi, np.eye(3), mtx, (IMAGE_W, IMAGE_H),
                                                             cv2.CV_32FC1, cv2.omnidir.RECTIFY_PERSPECTIVE)
        dst = cv2.remap(sample_test, map1, map2, interpolation = cv2.INTER_LINEAR, borderMode = cv2.BORDER_CONSTANT)
        # dst = cv2.fisheye.undistortImage(sample_test, mtx, dist, None, optim_matrix)

    cv2.namedWindow('original', 0)
    cv2.resizeWindow('original', 500, 500)
    cv2.imshow('original', sample_test)

    cv2.namedWindow('undistort', 0)
    cv2.resizeWindow('undistort', 500, 500)
    cv2.imshow('undistort', dst)

    if cv2.waitKey(0) == 27 & 0xff:
        cv2.destroyAllWindows()


def calibcamera_pinhole(objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (IMAGE_W, IMAGE_H), None, None,
                                                       criteria = criteria)
    calc_reproj_error(objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs)
    return ret, mtx, dist, rvecs, tvecs


def calibcamera_fisheye(objpoints, imgpoints):
    try:
        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, (IMAGE_W, IMAGE_H), None, None,
                                                             flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                                                                     # + cv2.fisheye.CALIB_CHECK_COND
                                                                     + cv2.fisheye.CALIB_FIX_SKEW,
                                                             criteria = criteria)
    except IOError:
        print("KB with flags-cv2.fisheye.CALIB_CHECK_COND failed! try using KB without flags "
              "flags-cv2.fisheye.CALIB_CHECK_COND")
        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(objpoints, imgpoints, (IMAGE_W, IMAGE_H), None, None,
                                                             flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                                                                     + cv2.fisheye.CALIB_FIX_SKEW,
                                                             criteria = criteria)

    calc_reproj_error(objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs)
    return ret, mtx, dist, rvecs, tvecs


def calibcamera_fisheye_mei(objpoints, imgpoints):
    ret, mtx, xi, dist, rvecs, tvecs, idx = cv2.omnidir.calibrate(objpoints, imgpoints, (IMAGE_W, IMAGE_H),
                                                                  None, None, None,
                                                                  flags = cv2.omnidir.CALIB_FIX_SKEW,
                                                                  criteria = criteria)
    calc_reproj_error(objpoints, imgpoints, ret, mtx, dist, rvecs, tvecs, xi, idx)
    return ret, mtx, xi, dist, rvecs, tvecs


def main():
    extra_image_path = "./calibrationdata"
    # save path
    save_result_path = args.result_path
    save_images_path = args.images_path

    # extra_data(extra_image_path)
    images = glob.glob('/home/ubuntu/Desktop/project/sensor-calibration/tmp/qirong_right_0701/*.jpg')
    objpoints, imgpoints = load_images(images)

    if not fisheye:
        print("----------------calibrate pinhole camera---------------------")
        ret, mtx, dist, rvecs, tvecs = calibcamera_pinhole(objpoints, imgpoints)
    elif fisheye:
        print("----------------calibrate fisheye camera---------------------")
        if args.fisheye_mode == "KB":
            ret, mtx, dist, rvecs, tvecs = calibcamera_fisheye(objpoints, imgpoints)
        elif args.fisheye_mode == "Mei":
            ret, mtx, xi, dist, rvecs, tvecs = calibcamera_fisheye_mei(objpoints, imgpoints)

    if args.fisheye_mode == "Mei":
        save_results(save_result_path, save_images_path, mtx, dist, xi)
    else:
        save_results(save_result_path, save_images_path, mtx, dist)

    if visualize_result:
        if args.fisheye_mode == "Mei":
            visualize_result(images, mtx, dist, xi)
        else:
            visualize_result(images, mtx, dist)


if __name__ == '__main__':
    main()
