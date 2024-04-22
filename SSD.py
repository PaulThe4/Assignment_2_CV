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

# Step 2: Select super-pixel patches on the frames
# For simplicity, let's select a pixel and define a patch around it
x, y = 100, 100  # Coordinates of the pixel
patch_size = 20  # Increased patch size

# Extract patches from the frames
patch1 = frame1[y:y+patch_size, x:x+patch_size]
patch2 = frame2[y:y+patch_size, x:x+patch_size]

# Step 3: Compute SIFT features for the patches
sift = cv2.SIFT_create()

# Compute SIFT keypoints and descriptors for patch 1
kp1, desc1 = sift.detectAndCompute(patch1, None)

# Compute SIFT keypoints and descriptors for patch 2
kp2, desc2 = sift.detectAndCompute(patch2, None)

# Step 4: Check the validity of keypoints and compute SSD
if kp1 and kp2 and desc1 is not None and desc2 is not None:
    # Print the number of keypoints detected
    print("Number of keypoints in patch 1:", len(kp1))
    print("Number of keypoints in patch 2:", len(kp2))

    # Compute SSD
    ssd = np.sum((desc1 - desc2) ** 2)
    print("SSD:", ssd)
else:
    print("Error: Unable to compute SIFT descriptors for one or both patches or no keypoints detected.")

# Optional: Visualize the patches and keypoints
frame1_with_keypoints = cv2.drawKeypoints(frame1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
frame2_with_keypoints = cv2.drawKeypoints(frame2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Patch 1", patch1)
cv2.imshow("Patch 2", patch2)
cv2.imshow("Frame 1 with Keypoints", frame1_with_keypoints)
cv2.imshow("Frame 2 with Keypoints", frame2_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()