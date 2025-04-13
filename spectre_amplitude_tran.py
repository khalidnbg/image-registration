import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from matplotlib.gridspec import GridSpec
import cv2  # Pour charger votre propre image

# 1. Charger votre image de référence
# Remplacez 'chemin_vers_votre_image.jpg' par le chemin de votre image
image_path = 'images/img_org_translated_img.PNG'
image_I = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Charger en niveaux de gris

# Vérifier si l'image a bien été chargée
if image_I is None:
    raise ValueError(f"Impossible de charger l'image: {image_path}")

# 2. Créer une version translatée de l'image
shift = (10, 15)  # Translation de 30 pixels vers le bas et 50 pixels vers la droite
image_J = ndimage.shift(image_I, shift, mode='wrap')

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
ax1.set_title('Image de référence (I)')
ax1.axis('off')

# Image J (translatée)
ax2 = plt.subplot(gs[0, 1])
ax2.imshow(image_J, cmap='gray')
ax2.set_title(f'Image translatée (J) - Shift {shift}')
ax2.axis('off')

# Spectre d'amplitude de I
ax3 = plt.subplot(gs[1, 0])
ax3.imshow(display_spectrum(spectrum_I), cmap='viridis')
ax3.set_title('Spectre d\'amplitude de I')
ax3.axis('off')

# Spectre d'amplitude de J
ax4 = plt.subplot(gs[1, 1])
ax4.imshow(display_spectrum(spectrum_J), cmap='viridis')
ax4.set_title('Spectre d\'amplitude de J')
ax4.axis('off')

plt.tight_layout()

# Vérification mathématique de l'invariance
diff = np.abs(spectrum_I - spectrum_J)
max_diff = np.max(diff)
mean_diff = np.mean(diff)
print(f"Différence maximale entre les spectres d'amplitude: {max_diff:.4e}")
print(f"Différence moyenne entre les spectres d'amplitude: {mean_diff:.4e}")

# Sauvegarder la figure pour le rapport
plt.savefig('invariance_translation_2D.png', dpi=300, bbox_inches='tight')
plt.show()