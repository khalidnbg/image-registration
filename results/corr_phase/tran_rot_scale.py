import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray
from skimage.filters import window, difference_of_gaussians
from skimage.transform import rotate, rescale, warp_polar, AffineTransform, warp
from skimage.registration import phase_cross_correlation
import numpy as np
from scipy.fft import fft2, fftshift

# ========================
# Charger tes propres images
# ========================
image_path = "results/brain.jpg"
transformed_path = "results/brain_transformed.jpg"

def load_as_gray(path):
    img = io.imread(path)
    if img.ndim == 3:
        img = rgb2gray(img)
    return img

image = load_as_gray(image_path)
rts_image = load_as_gray(transformed_path)

# ========================
# Ajouter un padding pour éviter les pertes de bord
# ========================
pad_width = max(image.shape)
image_pad = np.pad(image, pad_width, mode='constant')
rts_pad = np.pad(rts_image, pad_width, mode='constant')

# --- PARTIE 1 : log-polaire direct (test) ---
radius = min(image_pad.shape) // 2
warped_image = warp_polar(image_pad, radius=radius, scaling="log")
warped_rts = warp_polar(rts_pad, radius=radius, scaling="log")

shifts, error, phasediff = phase_cross_correlation(
    warped_image, warped_rts, upsample_factor=20, normalization=None
)
shiftr_est, shiftc_est = shifts[:2]
klog = radius / np.log(radius)
shift_scale_est = 1 / (np.exp(shiftc_est / klog))

print("----- Partie 1 -----")
print(f"Rotation estimée (°) : {shiftr_est}")
print(f"Échelle estimée : {shift_scale_est:.4f}")

# --- PARTIE 2 : FFT magnitude + log-polaire ---
image_filt = difference_of_gaussians(image_pad, 5, 20)
rts_filt = difference_of_gaussians(rts_pad, 5, 20)

wimage = image_filt * window("hann", image_pad.shape)
rts_wimage = rts_filt * window("hann", rts_pad.shape)

image_fs = np.abs(fftshift(fft2(wimage)))
rts_fs = np.abs(fftshift(fft2(rts_wimage)))

shape = image_fs.shape
radius_fft = shape[0] // 8

warped_image_fs = warp_polar(image_fs, radius=radius_fft, output_shape=shape, scaling="log", order=0)
warped_rts_fs = warp_polar(rts_fs, radius=radius_fft, output_shape=shape, scaling="log", order=0)

warped_image_fs = warped_image_fs[: shape[0] // 2, :]
warped_rts_fs = warped_rts_fs[: shape[0] // 2, :]

shifts_fft, error_fft, phasediff_fft = phase_cross_correlation(
    warped_image_fs, warped_rts_fs, upsample_factor=10, normalization=None
)
shiftr_fft, shiftc_fft = shifts_fft[:2]

recovered_angle = (360 / shape[0]) * shiftr_fft
klog_fft = shape[1] / np.log(radius_fft)
recovered_scale = np.exp(shiftc_fft / klog_fft)

print("----- Partie 2 -----")
print(f"Rotation estimée (°) : {recovered_angle:.4f}")
print(f"Échelle estimée : {recovered_scale:.4f}")

# --- PARTIE 3 : Estimation translation corrigée ---
theta_rad = -np.deg2rad(recovered_angle)
inv_scale = 1 / recovered_scale

# Appliquer l'inverse de rotation + échelle
inverse_transform = AffineTransform(scale=(inv_scale, inv_scale), rotation=theta_rad, translation=(0,0))
corrected_image = warp(rts_pad, inverse_transform.inverse, output_shape=image_pad.shape)

# Estimer la translation finale
shift, error, diffphase = phase_cross_correlation(image_pad, corrected_image, upsample_factor=20)

print("----- Partie 3 -----")
print(f"Translation estimée (y, x) : {shift}")

# --- Affichage final pour vérification ---
fig, axes = plt.subplots(1,3, figsize=(15,5))
axes[0].imshow(image_pad, cmap='gray')
axes[0].set_title("Image originale (padded)")
axes[1].imshow(rts_pad, cmap='gray')
axes[1].set_title("Image transformée (padded)")
axes[2].imshow(corrected_image, cmap='gray')
axes[2].set_title("Transformée corrigée (rotation+scale)")
plt.show()
