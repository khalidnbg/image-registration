import cv2
from skimage import data, transform
from skimage import img_as_ubyte
import numpy as np

# Lecture des deux images
img1 = data.camera()

angle = 45
scale = 0.8

# Rotation et redimensionnement
img2 = transform.rotate(img1, angle)
img2 = transform.rescale(img2, scale, channel_axis=-1 if img1.ndim == 3 else None)

# Conversion en uint8
img1_uint8 = img_as_ubyte(img1)
img2_uint8 = img_as_ubyte(img2)

sift = cv2.SIFT_create()

# Détecteur SIFT
kp1, des1 = sift.detectAndCompute(img1_uint8, None)
kp2, des2 = sift.detectAndCompute(img2_uint8, None)

# Appariement avec FLANN
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# Extraction des points correspondants
src_pts = [kp1[m.queryIdx].pt for m in good_matches]
dst_pts = [kp2[m.trainIdx].pt for m in good_matches]

# Estimation homographie avec RANSAC
H, mask = cv2.findHomography(
    np.float32(src_pts), np.float32(dst_pts),
    cv2.RANSAC, 5.0
)

# Affichage
img_matches = cv2.drawMatches(img1_uint8, kp1, img2_uint8, kp2, good_matches, None)
cv2.imshow("Correspondances", img_matches)
cv2.waitKey(0)
