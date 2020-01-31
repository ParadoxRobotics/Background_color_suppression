# Object retrieval + centroid with background removal and blob.
# object = justin bridou pack

# General and Vision lib
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

PI = 3.1415926535897932384626433832795

# camera intrinsic matrix :
IM = np.array([[2.1452726429748178e+04, 0.0, 2.9289569146473295e+03], [0.0, 2.1535401203902496e+04, 1.1405073747998595e+03], [0.0, 0.0, 1.0]])
FX = IM[0,0]
FY = IM[1,1]
CX = IM[0,2]
CY = IM[1,2]
cam_center = (CX, CY)

# camera distortion matrix :
DM = np.array([-2.0364959389954320e+00, -1.7895789638611801e+00, 5.0724202929402676e-02, -6.0574189297764064e-02, 1.5102713916250274e+00])

# load image in RGB space :
state_frame = cv2.imread("JB_7.png")

# undistort and resize
state_frame = cv2.undistort(state_frame, IM, DM)
state_frame = cv2.resize(state_frame, (640, 480))
state_frame_RGB = cv2.cvtColor(state_frame, cv2.COLOR_BGR2RGB)
# convert in HSV space and grayscale
state_frame_hsv = cv2.cvtColor(state_frame, cv2.COLOR_BGR2HSV)
state_frame_gray = cv2.cvtColor(state_frame, cv2.COLOR_BGR2GRAY)

# get ROI from image (background or usage)
roi = cv2.selectROI(state_frame)
cv2.destroyAllWindows()
# crop in HSV image space
crop = state_frame_hsv[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

# get the different HSV component for the crop and the state image
H_state, S_state, V_state = cv2.split(state_frame_hsv)
H_state_crop, S_state_crop, V_state_crop = cv2.split(crop)

# calculate STD/mean for cropped HSV plane
mH, stdH = cv2.meanStdDev(H_state_crop)
mS, stdS = cv2.meanStdDev(S_state_crop)
mV, stdV = cv2.meanStdDev(V_state_crop)

# calculate valid min max for HSV
minH = mH - stdH*3
maxH = mH + stdH*3
maskH = 255-cv2.inRange(H_state, minH, maxH)
"""
plt.imshow(maskH, cmap='gray')
plt.show()
"""

minS = mS - stdS*3
maxS = mS + stdS*3
maskS = 255-cv2.inRange(S_state, minS, maxS)
"""
plt.imshow(maskS, cmap='gray')
plt.show()state_frame_RGB
"""

minV = mV - stdV*3
maxV = mV + stdV*3
maskV = 255-cv2.inRange(V_state, minV, maxV)
"""
plt.imshow(maskV, cmap='gray')
plt.show()
"""

# print(minH, maxH, minS, maxS, minV, maxV)

# create bitwise_and mask on the grayscale state image
bitwise_H = cv2.bitwise_and(state_frame_gray, state_frame_gray, mask = maskH)
bitwise_S = cv2.bitwise_and(state_frame_gray, state_frame_gray, mask = maskS)
bitwise_V = cv2.bitwise_and(state_frame_gray, state_frame_gray, mask = maskV)


plt.imshow(bitwise_H, cmap='gray')
plt.show()
plt.imshow(bitwise_S, cmap='gray')
plt.show()
plt.imshow(bitwise_V, cmap='gray')
plt.show()


# merge multiple grayscale masked image with coeff
weight_H = 0.10
weight_S = 0.0
weight_V = 0.90

# result + normalization
result_gray = bitwise_H * weight_H + bitwise_S * weight_S + bitwise_V * weight_V
result_gray = cv2.normalize(result_gray, result_gray, 0, 255, cv2.NORM_MINMAX)
# plot
plt.imshow(result_gray, cmap='gray')
plt.show()

# apply Threshold for image binarization
ret, result_bin = cv2.threshold(result_gray, 1, 255, cv2.THRESH_BINARY)
# erosion + dilation = opening suppression
kernel_22 = np.ones((2,2),np.uint8)
kernel_33 = np.ones((3,3),np.uint8)
result_bin = cv2.morphologyEx(result_bin, cv2.MORPH_OPEN, kernel_22)
# CV_8U conversion
result_bin = np.uint8(result_bin)
# plot
plt.imshow(result_bin, cmap='gray')
plt.show()

# find geomtric contour with shape approximation
contours, hierarchy = cv2.findContours(result_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# approximate using rotated bounding rectangle

if len(contours) != 0:
    contour_max = max(contours, key = cv2.contourArea)
    rect = cv2.minAreaRect(contour_max)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(state_frame_RGB, [box], 0, (255,0,0), 4)

    # print pose + angle (pixel space)
    pose_x = rect[0][0]
    pose_y = rect[0][1]
    print("position : ", pose_x, pose_y)
    # calculate angle [0,180]
    if rect[1][0] < rect[1][1]:
        angle = 90-rect[2]
    else:
        angle = -rect[2]
    print("angle : ", angle)

    # draw center
    cv2.arrowedLine(state_frame_RGB, (int(pose_x), int(pose_y)), (int(pose_x + 90 * np.cos(angle*PI/180)), int(pose_y - 90 * np.sin(angle*PI/180))), (0,0,255), 3)
    cv2.arrowedLine(state_frame_RGB, (int(pose_x), int(pose_y)), (int(pose_x + 90 * np.cos(1.5708 + angle*PI/180)), int(pose_y - 90 * np.sin(1.5708 + angle*PI/180))), (0,0,255), 3)
    # plotor
    plt.imshow(state_frame_RGB)
    plt.show()

else:
    print("no object in the area")

for i in range(1,20):
    # load image in RGB space :
    i = str(i)
    state_frame = cv2.imread('JB_'+i+'.png')
    # undistort and resize
    state_frame = cv2.undistort(state_frame, IM, DM)
    state_frame = cv2.resize(state_frame, (640, 480))
    state_frame_RGB = cv2.cvtColor(state_frame, cv2.COLOR_BGR2RGB)
    # convert in HSV space and grayscale
    state_frame_hsv = cv2.cvtColor(state_frame, cv2.COLOR_BGR2HSV)
    state_frame_gray = cv2.cvtColor(state_frame, cv2.COLOR_BGR2GRAY)
    # split HSV space
    H_state, S_state, V_state = cv2.split(state_frame_hsv)
    # calculate mask
    maskH = 255-cv2.inRange(H_state, minH, maxH)
    maskS = 255-cv2.inRange(S_state, minS, maxS)
    maskV = 255-cv2.inRange(V_state, minV, maxV)
    # create bitwise_and mask on the grayscale state image
    bitwise_H = cv2.bitwise_and(state_frame_gray, state_frame_gray, mask = maskH)
    bitwise_S = cv2.bitwise_and(state_frame_gray, state_frame_gray, mask = maskS)
    bitwise_V = cv2.bitwise_and(state_frame_gray, state_frame_gray, mask = maskV)
    # result + normalization
    result_gray = bitwise_H * weight_H + bitwise_S * weight_S + bitwise_V * weight_V
    result_gray = cv2.normalize(result_gray, result_gray, 0, 255, cv2.NORM_MINMAX)
    # apply Threshold for image binarization
    ret, result_bin = cv2.threshold(result_gray, 1, 255, cv2.THRESH_BINARY)
    # erosion + dilation = opening suppression
    kernel_22 = np.ones((2,2),np.uint8)
    kernel_33 = np.ones((3,3),np.uint8)
    result_bin = cv2.morphologyEx(result_bin, cv2.MORPH_OPEN, kernel_22)
    # CV_8U conversion
    result_bin = np.uint8(result_bin)
    # find geomtric contour with shape approximationor
    contours, hierarchy = cv2.findContours(result_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # approximate using rotated bounding rectangle
    if len(contours) != 0:
        contour_max = max(contours, key = cv2.contourArea)
        rect = cv2.minAreaRect(contour_max)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(state_frame_RGB, [box], 0, (255,0,0), 4)
        # print pose + angle (pixel space)
        pose_x = rect[0][0]
        pose_y = rect[0][1]
        print("position : ", pose_x, pose_y)
        # calculate angle [0,180]
        if rect[1][0] < rect[1][1]:
            angle = 90-rect[2]
        else:
            angle = -rect[2]
        print("angle : ", angle)
        # draw center
        cv2.arrowedLine(state_frame_RGB, (int(pose_x), int(pose_y)), (int(pose_x + 80 * np.cos(angle*PI/180)), int(pose_y - 80 * np.sin(angle*PI/180))), (0,0,255), 3)
        cv2.arrowedLine(state_frame_RGB, (int(pose_x), int(pose_y)), (int(pose_x + 80 * np.cos(1.5708 + angle*PI/180)), int(pose_y - 80 * np.sin(1.5708 + angle*PI/180))), (0,0,255), 3)
        # plot
        plt.imshow(state_frame_RGB)
        plt.show()
    else:
        print("no object in the area")
