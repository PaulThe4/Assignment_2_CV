import cv2
import numpy as np

# Function to perform Harris corner detection manually
def harris_corner_detection(image_patch, threshold=0.1):
    # Calculate image gradients using Sobel operators
    sobel_x = cv2.Sobel(image_patch, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_patch, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute elements of the structure tensor
    Ixx = sobel_x ** 2
    Ixy = sobel_x * sobel_y
    Iyy = sobel_y ** 2
    
    # Compute sums of products of derivatives
    Sxx = cv2.GaussianBlur(Ixx, (3, 3), 0)
    Sxy = cv2.GaussianBlur(Ixy, (3, 3), 0)
    Syy = cv2.GaussianBlur(Iyy, (3, 3), 0)
    
    # Compute Harris corner response
    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy
    corner_response = det - 0.04 * (trace ** 2)
    
    # Threshold corner response to identify corners
    corners = np.where(corner_response > threshold * corner_response.max())
    
    return corners

# Load the video file
video_path = '/Users/sonipriyapaul/Downloads/Assignment_3_CV/Object_Video.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Unable to open the video file.")
    exit()

# Read a frame from the video
ret, frame = cap.read()

# Check if the frame was read successfully
if not ret:
    print("Error: Unable to read the frame.")
    exit()

# Display the frame
cv2.imshow('Frame', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Select a region of interest (ROI) with a corner
# You can manually select the coordinates of the ROI
roi_x, roi_y, roi_w, roi_h = 200, 400, 500, 1200
roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]

# Perform Harris corner detection on the selected ROI
corners_manual = harris_corner_detection(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

# Draw detected corners on the ROI
for corner in zip(*corners_manual[::-1]):
    cv2.circle(roi, corner, 5, (0, 255, 0), -1)

# Display the ROI with manually detected corners
cv2.imshow('Harris Corner detected with Manual Corners', roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Release the video capture object
cap.release()