import cv2
import numpy as np

# Step 1: Extract frames from the video
video_path = '/Users/sonipriyapaul/Downloads/Assignment_3_CV/Object_Video.mp4'
cap = cv2.VideoCapture(video_path)

# Extract frame at t=0 seconds
cap.set(cv2.CAP_PROP_POS_MSEC, 0)
ret, frame1 = cap.read()

# Extract frame at t=2 seconds
cap.set(cv2.CAP_PROP_POS_MSEC, 2000)
ret, frame2 = cap.read()

# Release the video capture object
cap.release()

#im_src = cv2.imread(frame1)

# Four corners of the book in source image
pts_src = np.array([[410, 510], [623, 510], [370, 1460],[670, 1460]])
 
# Read destination image.
#im_dst = cv2.imread(frame2)
# Four corners of the book in destination image.
pts_dst = np.array([[315, 510],[530, 510],[280, 1460],[580, 1460]])

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)

# Warp source image to destination based on homography
im_out = cv2.warpPerspective(frame1, h, (frame2.shape[1],frame2.shape[0]))

# Display images
cv2.imshow("Source Image", frame1)
cv2.imshow("Destination Image", frame2)
cv2.imshow("Warped Source Image", im_out)

cv2.waitKey(0)

# Compute the inverse homography matrix
H_inv = np.linalg.inv(h)

print("Homography Matrix:")
print(h)
print("\nInverse Homography Matrix:")
print(H_inv)

# Define the filename for the text file
output_file = "/Users/sonipriyapaul/Downloads/Assignment_2_CV/homography_matrices.txt"

# Write the matrices to the text file
with open(output_file, 'w') as f:
    f.write("Homography Matrix:\n")
    f.write(np.array2string(h))
    f.write("\n\nInverse Homography Matrix:\n")
    f.write(np.array2string(H_inv))

print("Homography matrices have been saved to", output_file)