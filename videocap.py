## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
## Open CV and Numpy integration ##
###############################################

import pyrealsense2 as rs
import numpy as np
import cv2
import pickle as pkl

depth_width = 1024
depth_height = 768
color_width = 1280
color_height = 720
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, 30)
config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

last_point=None

def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global last_point

    if event == cv2.EVENT_LBUTTONDOWN:
        last_point = (x, y)

cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback("frame", click_and_crop)

try:
    while True:
        print()
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        # Create 3D Image
        pc = rs.pointcloud()
        points = pc.calculate(depth_frame)
        v, t = points.get_vertices(), points.get_texture_coordinates()
        verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)
        texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)
        x0 = verts[:, 0].flatten()
        y0 = verts[:, 1].flatten()
        z0 = verts[:, 2].flatten()
        x0= x0.reshape(depth_height, depth_width)
        y0= y0.reshape(depth_height, depth_width)
        z0= z0.reshape(depth_height, depth_width)
        image_3d = cv2.merge((x0, y0, z0))
        image2save = image_3d.copy()
        image_3d[np.where(np.logical_or(image_3d[:, :, 2] <= 0.498, image_3d[:, :, 2] >= 0.55))] = [0, 0, 0]
        img = cv2.normalize(src=image_3d, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,
                            dtype=cv2.CV_8U)
        #gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        #blur = cv2.GaussianBlur(depth_image, (5, 5), 0)

        im_bw = cv2.Canny(img, 100, 220)
        contours, hierarchy = cv2.findContours(im_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #for c in contours:
        #    x, y, w, h = cv2.boundingRect(c)
        #    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2)

            # get the min area rect
           # rect = cv2.minAreaRect(c)
            #box = cv2.boxPoints(rect)

        c = max(contours, key=cv2.contourArea)
        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image
        cv2.drawContours(image_3d, [c], -1, (0, 255, 0), 3)
        cv2.circle(image_3d, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(image_3d, "center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # Display the resulting fr
        #X,Y
        #goal = image_3d[X,Y,:]

        if last_point != None:
            print(image_3d[last_point])

        cv2.imshow('frame', image_3d)
        #color_frame, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # Stack both images horizontally
        # images = np.hstack((color_image, depth_colormap))
        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense_depth', depth_colormap)
        #cv2.imshow('RealSense_RGB', color_image)
        #cv2.imshow('image_3d', image_3d)
        key=cv2.waitKey(1)
        if key == 99:
            with open("Im_345", 'wb+') as f:
                # Step 3
                pkl.dump(image2save, f)
        # Press esc or 'q' to close the image window

finally:
    # Stop streaming
    pipeline.stop()