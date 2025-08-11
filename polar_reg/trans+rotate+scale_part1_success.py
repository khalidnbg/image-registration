import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.color import rgb2gray
from skimage.filters import window, difference_of_gaussians
from skimage.transform import rotate, rescale, warp_polar
from skimage.registration import phase_cross_correlation
from scipy.fft import fft2, fftshift

# Paramètres de transformation
angle = 24        # rotation en degrés
scale = 1.4       # facteur d'échelle
shiftr = 30       # translation verticale (pixels)
shiftc = 15       # translation horizontale (pixels)

# Chargement et préparation de l'image
image = rgb2gray(data.retina())

# Appliquer translation, rotation et mise à l'échelle dans cet ordre
translated = image[shiftr:, shiftc:]  # translation par découpage simple
rotated = rotate(translated, angle)
rescaled = rescale(rotated, scale, anti_aliasing=True)
sizer, sizec = image.shape
rts_image = rescaled[:sizer, :sizec]  # recadrage pour taille constante

# --- PARTIE 1 : tentative simple de registration log-polaire (échec si pas même centre) ---
radius = 705
warped_image = warp_polar(image, radius=radius, scaling="log")
warped_rts = warp_polar(rts_image, radius=radius, scaling="log")
shifts, error, phasediff = phase_cross_correlation(
    warped_image, warped_rts, upsample_factor=20, normalization=None
)
shiftr_est, shiftc_est = shifts[:2]
klog = radius / np.log(radius)
shift_scale_est = 1 / (np.exp(shiftc_est / klog))

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()
ax[0].set_title("Image originale")
ax[0].imshow(image, cmap='gray')
ax[1].set_title("Image transformée (T, R, S)")
ax[1].imshow(rts_image, cmap='gray')
ax[2].set_title("Log-polaire (original)")
ax[2].imshow(warped_image)
ax[3].set_title("Log-polaire (transformée)")
ax[3].imshow(warped_rts)
fig.suptitle("Échec de la registration log-polaire sans centre partagé")
plt.show()

print("----- Partie 1 -----")
print(f"Rotation attendue (°) : {angle}")
print(f"Rotation estimée (°) : {shiftr_est}")
print(f"Échelle attendue : {scale}")
print(f"Échelle estimée : {shift_scale_est:.4f}")
print()

# --- PARTIE 2 : registration rotation + échelle sur spectres de magnitude FFT ---

# Filtrage passe-bande
image_filt = difference_of_gaussians(image, 5, 20)
rts_filt = difference_of_gaussians(rts_image, 5, 20)

# Application fenêtre de Hann
wimage = image_filt * window('hann', image.shape)
rts_wimage = rts_filt * window('hann', image.shape)

# FFT et magnitude centrée
image_fs = np.abs(fftshift(fft2(wimage)))
rts_fs = np.abs(fftshift(fft2(rts_wimage)))

# Paramètres pour la transformation log-polaire sur FFT
shape = image_fs.shape
radius_fft = shape[0] // 8  # fréquence basse uniquement

# Transformations log-polaires sur magnitude FFT
warped_image_fs = warp_polar(
    image_fs, radius=radius_fft, output_shape=shape, scaling='log', order=0
)
warped_rts_fs = warp_polar(
    rts_fs, radius=radius_fft, output_shape=shape, scaling='log', order=0
)

# On conserve uniquement la moitié basse (fréquences)
warped_image_fs = warped_image_fs[: shape[0] // 2, :]
warped_rts_fs = warped_rts_fs[: shape[0] // 2, :]

# Phase correlation pour rotation + échelle
shifts_fft, error_fft, phasediff_fft = phase_cross_correlation(
    warped_image_fs, warped_rts_fs, upsample_factor=10, normalization=None
)
shiftr_fft, shiftc_fft = shifts_fft[:2]

# Conversion en rotation en degrés et facteur d'échelle
recovered_angle = (360 / shape[0]) * shiftr_fft
klog_fft = shape[1] / np.log(radius_fft)
recovered_scale = np.exp(shiftc_fft / klog_fft)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()
center = np.array(shape) // 2
ax[0].set_title("FFT magnitude (original, zoom)")
ax[0].imshow(
    image_fs[
        center[0] - radius_fft : center[0] + radius_fft,
        center[1] - radius_fft : center[1] + radius_fft,
    ],
    cmap='magma',
)
ax[1].set_title("FFT magnitude (transformée, zoom)")
ax[1].imshow(
    rts_fs[
        center[0] - radius_fft : center[0] + radius_fft,
        center[1] - radius_fft : center[1] + radius_fft,
    ],
    cmap='magma',
)
ax[2].set_title("FFT log-polaire (original)")
ax[2].imshow(warped_image_fs, cmap='magma')
ax[3].set_title("FFT log-polaire (transformée)")
ax[3].imshow(warped_rts_fs, cmap='magma')
fig.suptitle("Estimation rotation + échelle sur FFT")
plt.show()

print("----- Partie 2 -----")
print(f"Rotation attendue (°) : {angle}")
print(f"Rotation estimée (°) : {recovered_angle:.4f}")
print(f"Échelle attendue : {scale}")
print(f"Échelle estimée : {recovered_scale:.4f}")
print()

# --- PARTIE 3 : estimation de la translation corrigée ---

# Appliquer rotation et échelle inverse à l'image transformée
from skimage.transform import AffineTransform, warp

# Calculer matrice inverse de rotation + échelle
theta_rad = -np.deg2rad(recovered_angle)
inv_scale = 1 / recovered_scale

inverse_transform = AffineTransform(
    scale=(inv_scale, inv_scale),
    rotation=theta_rad,
    translation=(0, 0)
)

# Corriger l'image transformée
corrected_image = warp(rts_image, inverse_transform.inverse, output_shape=image.shape)

# Maintenant estimer la translation entre image originale et image corrigée
shift, error, diffphase = phase_cross_correlation(image, corrected_image, upsample_factor=20)

print("----- Partie 3 -----")
print(f"Translation estimée après correction rotation + échelle : {shift}")
