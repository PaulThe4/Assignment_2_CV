import cv2 # type: ignore
import numpy as np # type: ignore
import random

# Extract frames from the video and append them to frames list
cap = cv2.VideoCapture('/Users/sonipriyapaul/Downloads/Assignment_3_CV/Object_Video.mp4')
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Display the frame
    cv2.imshow('Video', frame)

    # Appending frames here
    frames.append(frame)

    # User will select ROI (press 's' to stop the video and select)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        # Select ROI using mouse drag and then click enter to lock that region in for roi selection process
        roi = cv2.selectROI('Select ROI', frame, fromCenter=False)
        cap.release()
        cv2.destroyAllWindows()
        break

# Choose an image frame
selected_frame = random.choice(frames)

# Identify a region of interest (ROI)
x, y, w, h = roi
roi_frame = selected_frame[y:y+h, x:x+w]

# Save the ROI as a separate image
cv2.imwrite('/Users/sonipriyapaul/Downloads/Assignment_3_CV/selected_roi.jpg', roi_frame)

# Convert the ROI to grayscale
gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

# Load the image from the dataset
image = cv2.imread('/Users/sonipriyapaul/Downloads/Assignment_3_CV/selected_roi.jpg')

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Create feature detector and descriptor
sift = cv2.SIFT_create()

# Detect and compute keypoints and descriptors
kp1, des1 = sift.detectAndCompute(gray_roi, None)
kp2, des2 = sift.detectAndCompute(gray_image, None)

# Create a BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors
matches = bf.match(des1, des2)

# Sort matches by distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw top matches
img_matches = cv2.drawMatches(roi_frame, kp1, image, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
cv2.imshow('Matches', img_matches)
cv2.imwrite('/Users/sonipriyapaul/Downloads/Assignment_3_CV/Matches.jpg', img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()