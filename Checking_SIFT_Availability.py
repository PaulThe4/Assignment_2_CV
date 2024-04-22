import cv2

# Print OpenCV version information
print("OpenCV version:", cv2.__version__)

# Check if SIFT is available
sift = cv2.SIFT_create()
if sift is None:
    print("SIFT is not available.")
else:
    print("SIFT is available.")