import cv2
import numpy as np

# Function to compute the integral image manually
def compute_integral_image(image):
    # Initialize an empty array for the integral image
    integral_image = np.zeros_like(image, dtype=np.float32)

    # Compute the cumulative sum over rows for each column separately
    for col in range(image.shape[1]):
        for row in range(image.shape[0]):
            integral_image[row, col] = image[:row+1, col].sum()

    # Compute the cumulative sum over columns for each row separately
    for row in range(integral_image.shape[0]):
        for col in range(1, integral_image.shape[1]):
            integral_image[row, col] += integral_image[row, col-1]

    return integral_image

# Video capture
cap = cv2.VideoCapture('/Users/sonipriyapaul/Downloads/Assignment_3_CV/Object_Video.mp4')  # Use 0 for webcam, or provide the path to a video file

# Read a frame from the video feed
ret, frame = cap.read()

# Convert the RGB image to grayscale
gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Compute integral image
integral_image = compute_integral_image(gray_image)

# Normalize integral image for display
#integral_image = cv2.normalize(integral_image, None, 0, 255, cv2.NORM_MINMAX)

# Display both RGB feed and integral image feed
cv2.imshow('RGB Feed', frame)
cv2.imshow('Integral Image Feed', integral_image.astype(np.uint8))

# Release video capture and close windows
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()