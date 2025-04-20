import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale images
img1 = cv2.imread('images/img_org_translated_img.PNG', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('images/transformed_image.png', cv2.IMREAD_GRAYSCALE)

def preprocess(img):
    # Pad image to prevent cropping
    h, w = img.shape
    pad = max(h, w)
    img_padded = np.zeros((pad*2, pad*2), dtype=np.uint8)
    img_padded[pad//2:pad//2 + h, pad//2:pad//2 + w] = img
    return img_padded

img1 = preprocess(img1)
img2 = preprocess(img2)

def get_fft_log_magnitude(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    mag = np.log1p(mag)
    return mag / np.max(mag)

def get_log_polar(img):
    h, w = img.shape
    center = (w // 2, h // 2)
    radius = np.hypot(center[0], center[1])
    log_polar = cv2.logPolar(img, center, radius / np.log(radius), cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    return log_polar

def estimate_rotation_scale(img1, img2):
    mag1 = get_fft_log_magnitude(img1)
    mag2 = get_fft_log_magnitude(img2)
    logpolar1 = get_log_polar(mag1)
    logpolar2 = get_log_polar(mag2)
    shift, _ = cv2.phaseCorrelate(logpolar1.astype(np.float32), logpolar2.astype(np.float32))
    angle = 360.0 * shift[1] / logpolar1.shape[0]
    scale = np.exp(shift[0] / logpolar1.shape[1])
    return angle, scale, logpolar1, logpolar2

angle, scale, logpolar1, logpolar2 = estimate_rotation_scale(img1, img2)

print(f"[INFO] Estimated Rotation: {angle:.2f} degrees")
print(f"[INFO] Estimated Scale: {scale:.4f}")

# Correct rotation and scale
(h, w) = img2.shape
center = (w // 2, h // 2)
M_rs = cv2.getRotationMatrix2D(center, -angle, 1.0 / scale)
img2_corrected = cv2.warpAffine(img2, M_rs, (w, h))

# Estimate translation
shift, _ = cv2.phaseCorrelate(img1.astype(np.float32), img2_corrected.astype(np.float32))
dx, dy = shift
print(f"[INFO] Estimated Translation: dx={dx:.2f}, dy={dy:.2f}")

# Apply translation
M_translate = np.float32([[1, 0, -dx], [0, 1, -dy]])
img2_registered = cv2.warpAffine(img2_corrected, M_translate, (w, h))

# Visualization
fig, axs = plt.subplots(2, 3, figsize=(16, 10))
axs[0, 0].imshow(img1, cmap='gray')
axs[0, 0].set_title("Reference Image")
axs[0, 0].axis('off')

axs[0, 1].imshow(img2, cmap='gray')
axs[0, 1].set_title("Input Image (Transformed)")
axs[0, 1].axis('off')

axs[0, 2].imshow(logpolar2, cmap='gray')
axs[0, 2].set_title("Log-Polar Transform")
axs[0, 2].axis('off')

axs[1, 0].imshow(img2_corrected, cmap='gray')
axs[1, 0].set_title("After Rotation + Scale Correction")
axs[1, 0].axis('off')

axs[1, 1].imshow(img2_registered, cmap='gray')
axs[1, 1].set_title("Final Registered Image")
axs[1, 1].axis('off')

diff = cv2.absdiff(img1, img2_registered)
axs[1, 2].imshow(diff, cmap='hot')
axs[1, 2].set_title("Difference Map")
axs[1, 2].axis('off')

plt.tight_layout()
plt.show()
