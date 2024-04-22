import cv2
import numpy as np

def image_stitch(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Initialize the feature detector and extractor (e.g., SIFT)
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors for both images
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Initialize the feature matcher using brute-force matching
    bf = cv2.BFMatcher()

    # Match the descriptors using brute-force matching
    matches = bf.match(descriptors1, descriptors2)

    # Select the top N matches
    num_matches = 50
    matches = sorted(matches, key=lambda x: x.distance)[:num_matches]

    # Extract matching keypoints
    src_points = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_points = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Estimate the homography matrix
    homography, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

    # Warp the first image using the homography
    result = cv2.warpPerspective(image1, homography, (image2.shape[1], image2.shape[0]))

    # Blending the warped image with the second image using alpha blending
    alpha = 0.5  # blending factor
    blended_image = cv2.addWeighted(result, alpha, image2, 1 - alpha, 0)

    # Return the blended image
    return blended_image
    

# Video capture
cap = cv2.VideoCapture('/Users/sonipriyapaul/Downloads/Assignment_3_CV/Object_Video.mp4')  # Use 0 for webcam, or provide the path to a video file

# Extract frame at t=0 seconds
cap.set(cv2.CAP_PROP_POS_MSEC, 0)
ret, frame1 = cap.read()

# Extract frame at t=7 seconds
cap.set(cv2.CAP_PROP_POS_MSEC, 4000)
ret, frame2 = cap.read()

cap.release()

# Stitch current frame with previous frame
result = image_stitch(frame1, frame2)

# Update previous frame with current frame
frame1 = frame2.copy()

# Display the stitched result
cv2.imshow('Panoramic Output', result)

# Release video capture and close window
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()