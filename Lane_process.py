import os
import numpy as np
import cv2
import pickle
import glob
from moviepy.editor import VideoFileClip
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

CAMERA_PARAMETERS_FILE = "parameter_camera.pkl"
WARP_PARAMETERS_FILE = "parameter_warp.pkl"
ret, mtx, dist, rvecs, tvecs = (None, None, None, None, None)

# undistort image
def undistort_image(img):
    global ret, mtx, dist, rvecs, tvecs
    if mtx is None or dist is None:
        # try to load from file
        try:
            camera_pickle = pickle.load(open(CAMERA_PARAMETERS_FILE, "rb"))
            (ret, mtx, dist, rvecs, tvecs) = camera_pickle
        except:
            calibrate_camera('camera_cal')
    return cv2.undistort(img, mtx, dist, None, mtx)

# Calibrate camera using the OpenCv chessboad method
def calibrate_camera(folder, nx=9, ny=6, show_corners=False):
    # prepare object points
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    objpoints = []
    imgpoints = []

    for fname in os.listdir(folder):
        print(fname)
        img = cv2.imread(folder + '/' + fname)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        # If found, draw corners
        if ret == True:
            #Append corners and object
            objpoints.append(objp)
            imgpoints.append(corners)
            if show_corners:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                plt.imshow(img)
                plt.savefig('output_images/corners.png', dpi=100)
                plt.show()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    output = open(CAMERA_PARAMETERS_FILE, 'wb')
    pickle.dump((ret, mtx, dist, rvecs, tvecs), output)
    output.close()


# Read Warp  parameter based on src_points and dst_points
def warpImageParameters(src_points, dst_points):
    Mw = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    return Mw, Minv


# Convert to HLS color space
def hls_color_thresh(img, threshLow, threshHigh):
    imgHLS = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # Return a binary image of threshold
    binary_output = np.zeros((img.shape[0], img.shape[1]))
    binary_output[
        (imgHLS[:, :, 0] >= threshLow[0]) & (imgHLS[:, :, 0] <= threshHigh[0]) & (imgHLS[:, :, 1] >= threshLow[1]) & (
                    imgHLS[:, :, 1] <= threshHigh[1]) & (imgHLS[:, :, 2] >= threshLow[2]) & (
                    imgHLS[:, :, 2] <= threshHigh[2])] = 1

    return binary_output


def sobel_x(img, sobel_kernel=3, min_thres=20, max_thres=100):
    # Apply the following steps to img
    # Convert to grayscale
    imghsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Channels L and S from HLS
    sobelx1 = cv2.Sobel(imghsl[:, :, 1], cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobelx2 = cv2.Sobel(imghsl[:, :, 2], cv2.CV_64F, 1, 0, ksize=sobel_kernel)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobelx1 = np.uint8(255 * sobelx1 / np.max(sobelx1))
    scaled_sobelx2 = np.uint8(255 * sobelx2 / np.max(sobelx2))

    # Create a binary mask where mag thresholds are met
    binary_outputx1 = np.zeros_like(scaled_sobelx1)
    binary_outputx1[(scaled_sobelx1 >= min_thres) & (scaled_sobelx1 <= max_thres)] = 1

    binary_outputx2 = np.zeros_like(scaled_sobelx2)
    binary_outputx2[(scaled_sobelx2 >= min_thres) & (scaled_sobelx2 <= max_thres)] = 1

    binary_output = np.zeros_like(scaled_sobelx1)
    binary_output[(binary_outputx1 == 1) | (binary_outputx2 == 1)] = 1
    # Return this mask as your binary_output image
    return binary_output


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * gradmag / np.max(gradmag))

    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    # Return this mask as your binary_output image
    return binary_output


# Direction threshold
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    absgraddir = np.arctan2(abs_sobely, abs_sobelx)

    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output


# Both Magnitude and direction threshold
def mag_dir_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), dir_thresh=(0, np.pi / 2)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Calculate the magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Calc angle
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)

    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * gradmag / np.max(gradmag))

    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1]) & (absgraddir >= dir_thresh[0]) & (
                absgraddir <= dir_thresh[1])] = 1

    return binary_output


def fitlines(binary_warped):
    # histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    #  Peak of the left and right histogram
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
                    nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(leftx) == 0:
        left_fit = []
    else:
        left_fit = np.polyfit(lefty, leftx, 2)

    if len(rightx) == 0:
        right_fit = []
    else:
        right_fit = np.polyfit(righty, rightx, 2)

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    if SHOW_IMAGE:
        show_image("Sliding window", out_img)
    return left_fit, right_fit, out_img



def fit_continuous(left_fit, right_fit, binary_warped):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
                nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
                (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
                    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial
    if len(leftx) == 0:
        left_fit_updated = []
    else:
        left_fit_updated = np.polyfit(lefty, leftx, 2)

    if len(rightx) == 0:
        right_fit_updated = []
    else:
        right_fit_updated = np.polyfit(righty, rightx, 2)

    return left_fit_updated, right_fit_updated


# find Curvature
def curvature(left_fit, right_fit, binary_warped):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    y_eval = np.max(ploty)

    ym_per_pix = 25 / 720  # meters per pixel
    xm_per_pix = 3.7 / 700  # meters per pixel

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit[0])
    center = (((left_fit[0] * 720 ** 2 + left_fit[1] * 720 + left_fit[2]) + (
                right_fit[0] * 720 ** 2 + right_fit[1] * 720 + right_fit[2])) / 2 - 640) * xm_per_pix

    return left_curverad, right_curverad, center


# Draw line and return image
def drawLine(undist, warped, left_fit, right_fit):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    # Fit new polynomials
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.polylines(color_warp, np.int32([pts_left]), color=(255, 0, 0), thickness=50, isClosed=False)
    cv2.polylines(color_warp, np.int32([pts_right]), color=(0, 0, 255), thickness=50, isClosed=False)

    # Warp the blank back to original image
    newwarp = cv2.warpPerspective(color_warp, Minv_persp, (color_warp.shape[1], color_warp.shape[0]))
    # Combine the result
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    return (result, color_warp)


def sanity_check(left_fit, right_fit, minSlope, maxSlope):
    # check if left and right fits exists
    # Calculates the tangent between left and right in two points, and check if it is in a reasonable threshold
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    if len(left_fit) == 0 or len(right_fit) == 0:
        status = False
        d0 = 0
        d1 = 0
    else:
        # Difference of slope
        L_0 = 2 * left_fit[0] * 460 + left_fit[1]
        R_0 = 2 * right_fit[0] * 460 + right_fit[1]
        d0 = np.abs(L_0 - R_0)

        L_1 = 2 * left_fit[0] * 720 + left_fit[1]
        R_1 = 2 * right_fit[0] * 720 + right_fit[1]
        d1 = np.abs(L_1 - R_1)

        if d0 >= minSlope and d0 <= maxSlope and d1 >= minSlope and d1 <= maxSlope:
            status = True
        else:
            status = False

    return (status, d0, d1)



def process_image(image):
    # Calibration arrays pre-calculated
    img_undist = undistort_image(image)
    if SHOW_IMAGE:
        show_image("Undistort Image", img_undist)

    global counter
    global ref_left
    global ref_right
    global left_fit
    global right_fit
    global M_persp
    global Minv_persp
    #test sample src and dst for transformation
    src = np.float32([[585, 450], [203, 720], [1127, 720], [695, 450]])
    dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])
    M_persp, Minv_persp = warpImageParameters(src, dst)

    # 2.Magnitude Threshold
    # Threshold color
    yellow_low = np.array([0, 100, 100])
    yellow_high = np.array([50, 255, 255])
    white_low = np.array([18, 0, 180])
    white_high = np.array([255, 80, 255])


    imgThres_yellow = hls_color_thresh(img_undist, yellow_low, yellow_high)
    imgThres_white = hls_color_thresh(img_undist, white_low, white_high)
    imgThr_sobelx = sobel_x(img_undist, 9, 80, 220)  # Sobel x

    img_mag_thr = np.zeros_like(imgThres_yellow)
    # imgThresColor[(imgThres_yellow==1) | (imgThres_white==1)] =1
    img_mag_thr[(imgThres_yellow == 1) | (imgThres_white == 1) | (imgThr_sobelx == 1)] = 1

    if SHOW_IMAGE:
        show_image("Combine Color and sobel Image",img_mag_thr)

    # 3. Birds-eye
    # Perspective array pre-calculated
    img_size = (img_mag_thr.shape[1], img_mag_thr.shape[0])
    binary_warped = cv2.warpPerspective(img_mag_thr, M_persp, img_size, flags=cv2.INTER_LINEAR)

    # 4. Detect lanes and return fit curves

    if counter == 0:
        left_fit, right_fit, out_imgfit = fitlines(binary_warped)
    else:
        left_fit, right_fit = fit_continuous(left_fit, right_fit, binary_warped)


    if SHOW_IMAGE:
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.imshow(binary_warped)
        plt.show()


    status_sanity, d0, d1 = sanity_check(left_fit, right_fit, 0, .55)

    # Calc curvature and center
    if status_sanity is True:
        # Save as last reliable fit
        ref_left, ref_right = left_fit, right_fit
        counter += 1
    else:  # Use the last realible fit
        left_fit, right_fit = ref_left, ref_right

    left_curv, right_curv, center_off = curvature(left_fit, right_fit, binary_warped)

    if SHOW_IMAGE:
        show_image("Warped Image", binary_warped)

    # Warp back to original and merge with image
    img_merge, img_birds = drawLine(img_undist, binary_warped, left_fit, right_fit)

    # Composition of images to final display
    img_out = np.zeros((576, 1280, 3), dtype=np.uint8)

    img_out[0:576, 0:1024, :] = cv2.resize(img_merge, (1024, 576))
    # b) Threshold
    img_out[0:288, 1024:1280, 0] = cv2.resize(img_mag_thr * 255, (256, 288))
    img_out[0:288, 1024:1280, 1] = cv2.resize(img_mag_thr * 255, (256, 288))
    img_out[0:288, 1024:1280, 2] = cv2.resize(img_mag_thr * 255, (256, 288))
    # c)Birds eye view
    img_out[310:576, 1024:1280, :] = cv2.resize(img_birds, (256, 266))

    # Write curvature and center in image
    TextL = "Left r: " + str(int(left_curv)) + " m"
    TextR = "Right r: " + str(int(right_curv)) + " m"
    TextC = "Center offset: " + str(round(center_off, 2)) + "m"
    fontScale = 1
    thickness = 2
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_out, TextL, (30, 40), fontFace, fontScale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    cv2.putText(img_out, TextR, (30, 70), fontFace, fontScale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    cv2.putText(img_out, TextC, (30, 100), fontFace, fontScale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    cv2.putText(img_out, "Threshold view", (1070, 30), fontFace, .8, (200, 200, 0), thickness, lineType=cv2.LINE_AA)
    cv2.putText(img_out, "Birds view", (1080, 305), fontFace, .8, (200, 200, 0), thickness, lineType=cv2.LINE_AA)
    if SHOW_IMAGE:
        show_image("final Image", img_out)
    return img_out


def show_image(title, img):
    f, axes = plt.subplots(1, 1, figsize=(30, 30))
    axes.set_title(title, fontsize=20)
    if len(img.shape) > 2:
        axes.imshow(img)
    else:
        #Gray image
        axes.imshow(img, cmap='gray')
    plt.show()


if __name__ == "__main__":
    # init flags
    global counter, SHOW_IMAGE
    counter = 0
    SHOW_IMAGE = False

    #test calibration
    img = cv2.imread("camera_cal/calibration1.jpg")
    img_undist = undistort_image(img)
    if SHOW_IMAGE:
        show_image("Undistort image example", img_undist)

    # test image file
    img = cv2.imread("test_images/test6.jpg")
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = process_image(imgRGB)

    #test video file
    if SHOW_IMAGE is False:
        counter=0
        output = 'project_video_output.mp4'
        #verified ok for previous off track part
        #clip1 = VideoFileClip("project_video.mp4").subclip(34, 43)
        # complete test
        clip1 = VideoFileClip("project_video.mp4")
        out_clip = clip1.fl_image(process_image)
        out_clip.write_videofile(output, audio=False)














