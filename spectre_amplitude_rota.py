import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from matplotlib.gridspec import GridSpec

# 1. Charger votre image de référence
# Remplacez 'chemin_vers_votre_image.jpg' par le chemin de votre image
image_path = 'images/img_org_translated_img.PNG'
image_I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Charger en niveaux de gris

# Vérifier si l'image a bien été chargée
if image_I is None:
    raise ValueError(f"Impossible de charger l'image: {image_path}")

# 2. Créer une version pivotée de l'image
angle = 45  # Rotation de 45 degrés
image_J = ndimage.rotate(image_I, angle, reshape=False, mode='constant', cval=0)

# 3. Calculer la transformée de Fourier des deux images
fft_I = np.fft.fft2(image_I)
fft_J = np.fft.fft2(image_J)

# 4. Calculer les spectres d'amplitude (en prenant le module des transformées)
spectrum_I = np.abs(fft_I)
spectrum_J = np.abs(fft_J)

# 5. Fonction pour afficher correctement les spectres d'amplitude (centrage et passage en log)
def display_spectrum(spectrum):
    # Centrer le spectre pour une meilleure visualisation
    spectrum_centered = np.fft.fftshift(spectrum)
    # Appliquer une transformation logarithmique pour mieux visualiser
    # (ajouter 1 pour éviter log(0))
    return np.log1p(spectrum_centered)

# 6. Préparation de l'affichage des 4 images demandées
plt.figure(figsize=(14, 10))
gs = GridSpec(2, 2, figure=plt.gcf())

# Image I (référence)
ax1 = plt.subplot(gs[0, 0])
ax1.imshow(image_I, cmap='gray')
ax1.set_title('Image originale')
ax1.axis('off')

# Image J (pivotée)
ax2 = plt.subplot(gs[0, 1])
ax2.imshow(image_J, cmap='gray')
ax2.set_title(f'Image pivotée de {angle}°')
ax2.axis('off')

# Spectre d'amplitude de I
ax3 = plt.subplot(gs[1, 0])
ax3.imshow(display_spectrum(spectrum_I), cmap='viridis')
ax3.set_title('Spectre d\'amplitude de l\'image originale')
ax3.axis('off')

# Spectre d'amplitude de J
ax4 = plt.subplot(gs[1, 1])
ax4.imshow(display_spectrum(spectrum_J), cmap='viridis')
ax4.set_title(f'Spectre d\'amplitude de l\'image pivotée de {angle}°')
ax4.axis('off')

plt.tight_layout()

# Sauvegarder la figure
plt.savefig('effet_rotation_spectre.png', dpi=300, bbox_inches='tight')
plt.show()

# Pour une comparaison plus détaillée, créons une figure supplémentaire qui montre plusieurs angles de rotation
angles = [0, 15, 30, 45, 60, 90]
plt.figure(figsize=(15, 12))

# Création d'une grille pour afficher 6 angles différents
for i, angle in enumerate(angles):
    # Rotation de l'image
    rotated_img = ndimage.rotate(image_I, angle, reshape=False, mode='constant', cval=0)
    
    # Calcul du spectre d'amplitude
    fft_rotated = np.fft.fft2(rotated_img)
    spectrum_rotated = np.abs(fft_rotated)
    
    # Affichage de l'image pivotée
    plt.subplot(3, 4, 2*i + 1)
    plt.imshow(rotated_img, cmap='gray')
    plt.title(f'Image pivotée de {angle}°')
    plt.axis('off')
    
    # Affichage du spectre d'amplitude correspondant
    plt.subplot(3, 4, 2*i + 2)
    plt.imshow(display_spectrum(spectrum_rotated), cmap='viridis')
    plt.title(f'Spectre d\'amplitude ({angle}°)')
    plt.axis('off')

plt.tight_layout()
plt.savefig('comparaison_rotations_spectres.png', dpi=300, bbox_inches='tight')
plt.show()