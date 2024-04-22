import cv2 as cv
import numpy as np

# Read the video file
video_path = '/Users/sonipriyapaul/Downloads/Assignment_3_CV/Object_Video.mp4'
cap = cv.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

# Read the first frame
ret, frame = cap.read()

# Check if the frame was read successfully
if not ret:
    print("Error: Unable to read the frame.")
    exit()

# Display the first frame
cv.imshow('Frame', frame)
cv.waitKey(0)
cv.destroyAllWindows()

# Select a region of interest (ROI)
# For simplicity, we'll use hardcoded coordinates here. You can adjust these coordinates as needed.
roi_x, roi_y, roi_w, roi_h = 200, 400, 500, 1200
roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

# Perform Canny edge detection on the ROI
canny_edges = cv.Canny(roi, 50, 150)

# Display the original ROI and the Canny edges
cv.imshow('ROI', roi)
cv.imshow('Canny Edges', canny_edges)
cv.waitKey(0)
cv.destroyAllWindows()

# Release the video capture object
cap.release()