import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float

# ===== PARAMÈTRES MODIFIABLES =====
# Changer ces valeurs pour tester différentes configurations

# Choix de l'image (décommentez celle que vous voulez utiliser)
# image = data.retina()          # Image de rétine (par défaut)
# image = data.coins()         # Pièces de monnaie
image = data.camera()        # Photo de cameraman
# image = data.astronaut()     # Photo d'astronaute
# image = data.coffee()        # Grains de café
# image = data.checkerboard()  # Damier
# image = data.brick()         # Texture de brique

# Angle de rotation à tester (en degrés)
angle = 28  # Changez cette valeur (ex: 15, 45, 90, 120, etc.)

# Rayon pour la transformation polaire (ajustez selon la taille de l'image)
radius = 705

# ===== TRAITEMENT =====
image = img_as_float(image)
rotated = rotate(image, angle)

# Déterminer si l'image a des canaux de couleur
if image.ndim == 3:
    # Image couleur (3D)
    image_polar = warp_polar(image, radius=radius, channel_axis=-1)
    rotated_polar = warp_polar(rotated, radius=radius, channel_axis=-1)
else:
    # Image en niveaux de gris (2D)
    image_polar = warp_polar(image, radius=radius)
    rotated_polar = warp_polar(rotated, radius=radius)

# ===== AFFICHAGE =====
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()

# Utiliser colormap appropriée pour les images en niveaux de gris
cmap = 'gray' if image.ndim == 2 else None

ax[0].set_title("Image Originale")
ax[0].imshow(image, cmap=cmap)
ax[0].axis('off')

ax[1].set_title(f"Image Tournée ({angle}°)")
ax[1].imshow(rotated, cmap=cmap)
ax[1].axis('off')

ax[2].set_title("Transformation Polaire - Original")
ax[2].imshow(image_polar, cmap=cmap)
ax[2].axis('off')

ax[3].set_title("Transformation Polaire - Tournée")
ax[3].imshow(rotated_polar, cmap=cmap)
ax[3].axis('off')

plt.tight_layout()
plt.show()

# ===== DÉTECTION DE LA ROTATION =====
shifts, error, phasediff = phase_cross_correlation(
    image_polar, rotated_polar, normalization=None
)

print(f'Type d\'image: {"Couleur" if image.ndim == 3 else "Niveaux de gris"}')
print(f'Dimensions de l\'image: {image.shape}')
print(f'Angle appliqué (sens anti-horaire): {angle}°')
print(f'Angle détecté: {shifts[0]:.2f}°')
print(f'Erreur: {error:.6f}')
print(f'Différence: {abs(angle - shifts[0]):.2f}°')

# ===== EXEMPLES D'AUTRES CONFIGURATIONS À TESTER ======
# 1. Image de pièces avec rotation de 45°:
#    image = data.coins()
#    angle = 45
#    radius = 200
# 
# 2. Image d'astronaute avec rotation de 90°:
#    image = data.astronaut()
#    angle = 90
#    radius = 300
# 
# 3. Damier avec rotation de 30°:
#    image = data.checkerboard()
#    angle = 30
#    radius = 150
# 
# 4. Grains de café avec rotation de 120°:
#    image = data.coffee()
#    angle = 120
#    radius = 250