import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data

def scale_image(img, scale_factor):
    """Met l’image à l’échelle tout en conservant sa taille d’origine par recadrage."""
    h, w = img.shape[:2]
    scaled_size = (int(w * scale_factor), int(h * scale_factor))
    scaled_img = cv2.resize(img, scaled_size, interpolation=cv2.INTER_LINEAR)

    # Recadrage ou padding pour ramener à la taille d’origine
    center = (scaled_img.shape[0] // 2, scaled_img.shape[1] // 2)
    start_y = center[0] - h//2
    start_x = center[1] - w//2
    canvas = scaled_img[start_y:start_y+h, start_x:start_x+w]
    return canvas

def estimate_scale_mellin(img1, img2):
    """Estime le facteur de mise à l’échelle entre deux images par transformée de Mellin."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Appliquer une fenêtre de Hanning
    hann = cv2.createHanningWindow((img1.shape[1], img1.shape[0]), cv2.CV_32F)
    img1 *= hann
    img2 *= hann

    # FFT et magnitude
    fft1 = np.fft.fft2(img1)
    fft2 = np.fft.fft2(img2)
    mag1 = np.fft.fftshift(np.log1p(np.abs(fft1)))
    mag2 = np.fft.fftshift(np.log1p(np.abs(fft2)))

    # Log-polar transform
    center = (img1.shape[1] // 2, img1.shape[0] // 2)
    M = img1.shape[0] / np.log(img1.shape[0] / 2)
    logpolar1 = cv2.logPolar(mag1, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)
    logpolar2 = cv2.logPolar(mag2, center, M, cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS)

    # Phase correlation pour décalage vertical = facteur d’échelle
    shift, _ = cv2.phaseCorrelate(logpolar1, logpolar2)
    scale = np.exp(shift[1] / logpolar1.shape[0])
    return scale

# === TEST avec une image réelle ===
true_scale = 1.5
base_img = data.camera()  # Image "camera" de scikit-image

# Conversion en uint8 pour compatibilité OpenCV
base_img = (base_img * 255).astype(np.uint8) if base_img.dtype == np.float64 else base_img

scaled_img = scale_image(base_img, true_scale)

estimated_scale = estimate_scale_mellin(base_img, scaled_img)

print(f"✔️ Facteur d’échelle réel     : {true_scale:.4f}")
print(f"📐 Facteur d’échelle estimé   : {estimated_scale:.4f}")

# Affichage
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Image originale")
plt.imshow(base_img, cmap='gray')
plt.subplot(1, 2, 2)
plt.title("Image mise à l’échelle")
plt.imshow(scaled_img, cmap='gray')
plt.tight_layout()
plt.show()