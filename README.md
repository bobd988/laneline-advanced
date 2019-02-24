## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort_cal_img.png "Undistorted"
[image2]: ./output_images/transformed_img.png "Road Transformed"
[image3]: ./output_images/combined_img.png "Binary Example"
[image4]: ./output_images/warped_img.png "Warp Example"
[image5]: ./output_images/fitline_img.png "Fit Visual"
[image6]: ./output_images/final_img.png "Output"
[image7]: ./output_images/slidingwindow_img.png "sliding window"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This project consists of a python script to detect lane lines from images/video. It uses opencv to identify lane lines and draw it over the original image. The script itself can work over one or a sequence of images (i.e. video file)
The source code is Lane_process.py and all functions are included. The main processing function is process_image() which is the actually pipeline function.  WHen SHOW_IMAGE is set to True from python code some intermediate images  will be during processing  and these were captured and saved in output_images folder. 
WHen SHOW_IMAGE is set to False from python code no intermediate images result will be displayed. Instead, the source video file will be processed  and video output file project_video_output.mp4 will be created.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located in calibrate_camera() function in Lane_process.py.  
The process is as below:

* prepare the objp[], which is same matrix as chessboard
* create list objpoints[] and imgpoints[] to hold the 3D and 2D pioints.
* for all calibration images in camera_cal use opencv calibrate_camera() to get mtx and dist, which are the distortions of the lenses


![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for this step is in undistort_image() function 

I first check if  pickle file exists or not. if pickle file exists it means calibration has been done.   If not program will perform  calibration  one and save camera parameters to a pickle file. After that cv2.undistort() to get the undistort image,the image showed below.
The image below shows one example result:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I tried several combinations of sobel gradient thresholds and color channel thresholds in multiple color spaces. If the difference in color between two points is very high, the derivative will be high. Below is an example of the combination of sobel magnitude and direction thresholds:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To implement perspective transformation I first need get the transformation parameter M and Minv. This is done by the warpImageParameters() function. After that I  fit a curve on the lane  and then unwarp to return to the original view.

Here I assume camera is mounted in a fixed position, and the relative position of the lanes are always the same.

The opencv function warp needs 4 origins and destinations points. After a few manual adjustments the 4 points are:

    src = np.float32([[585, 450], [203, 720], [1127, 720], [695, 450]])
    dst = np.float32([[320, 0], [320, 720], [960, 720], [960, 0]])

The image below demonstrates the results of the perspective transform:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In order to better estimate where the lane is I use a histogram on the bottom half of image. From this the left and right lane starting positions can be found.

Next I use sliding windows from bottom-up to identify non zero pixels for both left and right sides. 
![alt text][image7]
After identified all nonzero pixels use the numpy polyfit function to find the best second order polynomial to represent the lanes.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is given by following formula.

    Radius of curvature =​​ (1 + (dy/dx)**2)**1.5 / abs(d2y /dx2)

I need calculate the radius for both lines, left and right, and the chosen point is the base of vehicle, the bottom of image.

    x = ay2 + by + c

After calculating derivatives the formula is: 
    
    radius = (1 + (2a y_eval+b)**2)**1.5 / abs(2a)
    
Assuming the camera is mounted  in the center of the car then difference between the center of the image and the middle point of beginning of lines if the offset (in pixels). 
Note the above result are pixels and these need to convert from pixel to meters using the following conversions values:

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

The code is in curvature() function.

    left_curv, right_curv, center_off = curvature(left_fit, right_fit, binary_warped)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The opencv function polyfill is used to draw a area in the image and merge back to original perspective

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
To apply the pipeline on the video, you will need to  run the lane_process.py with SHOW_IMAGE manually set to False from python code.  Then process_image function will skip all image display and save to a video file project_video_output.mp4
The project video is project_video.mp4. The output video is  [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The solution shown above works  well for fixed situation where lane colors are yellow and white. It will not work well in situations were lane colors other than white and yellow. 
Also not robust enough when the curves are outside the chosen boundary region.  
The fixed parameters will make it not able to generalize the method.
Also there is performance issue with this pipeline. The above pipeline can only handle the images from video file. For real time camera video the pipeline must be able to process the images before the next image arrive. This will require a new design.