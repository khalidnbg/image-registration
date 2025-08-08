import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

# Charger l'image
image = cv2.imread('images/elephent.png', cv2.IMREAD_GRAYSCALE)

# --------- 1. Détection des contours (Canny) ---------
edges = cv2.Canny(image, 100, 200)

# --------- 2. Texture avec le filtre Laplacien ---------
laplacian = cv2.Laplacian(image, cv2.CV_64F)
laplacian = np.uint8(np.absolute(laplacian))

# --------- Affichage des résultats ---------
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.title('Image Grayscale')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title('Contours (Canny)')
plt.imshow(edges, cmap='gray')
plt.axis('off')

plt.subplot(2, 1, 2)
plt.title('Texture (Laplacien)')
plt.imshow(laplacian, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
