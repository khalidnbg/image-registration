import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import warp_polar
from skimage import data

# Fonction de transformation log-polaire
def log_polar_transform(image):
    """Transforme une image en coordonnées log-polaires"""
    center = (image.shape[1] // 2, image.shape[0] // 2)
    max_radius = min(center[0], center[1])
    
    # Appliquer la transformation log-polaire
    output_shape = (max_radius, 360)
    log_polar = warp_polar(image, center=center, radius=max_radius,
                           output_shape=output_shape, scaling='log')
    
    return log_polar

# Charger une image test (ou utiliser une image de démonstration)
try:
    image = cv2.imread('images/img_org_translated_img.PNG', 0)  # Charger en niveaux de gris
    if image is None:
        # Utiliser une image de démonstration si l'image n'est pas trouvée
        image = data.camera()
except:
    image = data.camera()

# Redimensionner pour avoir une taille raisonnable
image = cv2.resize(image, (256, 256))

# Créer des versions de l'image à différentes échelles
scale_factors = [0.5, 1.0, 2.0]
scaled_images = []
for scale in scale_factors:
    if scale == 1.0:
        scaled_images.append(image)
    else:
        new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
        scaled = cv2.resize(image, new_size)
        
        # Recadrer ou agrandir pour avoir la même taille que l'original
        result = np.zeros_like(image)
        
        # Calculer les coordonnées pour centrer l'image
        start_x = max(0, (result.shape[1] - scaled.shape[1]) // 2)
        start_y = max(0, (result.shape[0] - scaled.shape[0]) // 2)
        
        # Limites pour la découpe de l'image redimensionnée
        src_width = min(scaled.shape[1], result.shape[1])
        src_height = min(scaled.shape[0], result.shape[0])
        
        # Copier la partie centrale
        result[start_y:start_y+src_height, start_x:start_x+src_width] = \
            scaled[:src_height, :src_width]
        
        scaled_images.append(result)

# Appliquer la transformation log-polaire à chaque image
log_polar_images = [log_polar_transform(img) for img in scaled_images]

# Afficher les résultats
fig, axes = plt.subplots(len(scale_factors), 2, figsize=(10, 5*len(scale_factors)))

for i, scale in enumerate(scale_factors):
    # Image originale à l'échelle
    axes[i, 0].imshow(scaled_images[i], cmap='gray')
    axes[i, 0].set_title(f"Image à l'échelle x{scale}")
    axes[i, 0].axis('off')
    
    # Image log-polaire correspondante
    axes[i, 1].imshow(log_polar_images[i], cmap='gray')
    axes[i, 1].set_title(f"Transformation Log-Polaire (échelle x{scale})")
    axes[i, 1].axis('off')
    
    # Pour mettre en évidence la translation
    if i > 0:
        # Calculer la translation théorique
        log_shift = np.log(scale_factors[i]/scale_factors[0])
        theoretical_shift = int(log_shift * log_polar_images[0].shape[0])
        axes[i, 1].axhline(y=theoretical_shift, color='r', linestyle='--', 
                           alpha=0.5, label=f"Translation théorique: {theoretical_shift}px")
        axes[i, 1].legend()

plt.tight_layout()
plt.suptitle("Effet de la Transformation Log-Polaire (TLP) sur le facteur d'échelle", y=1.02)
plt.show()

# Visualiser la translation sur un graphique 1D
plt.figure(figsize=(10, 6))

for i, scale in enumerate(scale_factors):
    # Prendre une colonne représentative (milieu)
    column = log_polar_images[i][:, log_polar_images[i].shape[1]//2]
    plt.plot(column, label=f"Échelle x{scale}")
    
    # Calculer et afficher la translation théorique
    if i > 0:
        log_shift = np.log(scale_factors[i]/scale_factors[0])
        theoretical_shift = int(log_shift * log_polar_images[0].shape[0])
        plt.axvline(x=theoretical_shift, color=f'C{i}', linestyle='--', alpha=0.5,
                   label=f"Translation théorique pour échelle x{scale}")

plt.title("Profil d'intensité: translation dans le domaine log-polaire")
plt.xlabel("Position (pixels)")
plt.ylabel("Intensité")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()