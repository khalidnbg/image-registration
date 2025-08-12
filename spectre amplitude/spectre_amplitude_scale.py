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

# 2. Créer des versions mises à l'échelle de l'image
scale_factor = 0.5  # Réduction de 50%
# Redimensionner l'image (réduction)
height, width = image_I.shape
new_height, new_width = int(height * scale_factor), int(width * scale_factor)
image_reduced = cv2.resize(image_I, (new_width, new_height), interpolation=cv2.INTER_AREA)

# Redimensionner à nouveau pour comparer les spectres (même taille que l'original)
image_J = cv2.resize(image_reduced, (width, height), interpolation=cv2.INTER_LINEAR)

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

# Image J (mise à l'échelle)
ax2 = plt.subplot(gs[0, 1])
ax2.imshow(image_J, cmap='gray')
ax2.set_title(f'Image mise à l\'échelle (facteur {scale_factor}, puis retour à taille originale)')
ax2.axis('off')

# Spectre d'amplitude de I
ax3 = plt.subplot(gs[1, 0])
ax3.imshow(display_spectrum(spectrum_I), cmap='viridis')
ax3.set_title('Spectre d\'amplitude de l\'image originale')
ax3.axis('off')

# Spectre d'amplitude de J
ax4 = plt.subplot(gs[1, 1])
ax4.imshow(display_spectrum(spectrum_J), cmap='viridis')
ax4.set_title('Spectre d\'amplitude de l\'image mise à l\'échelle')
ax4.axis('off')

plt.tight_layout()

# Sauvegarder la figure
plt.savefig('effet_echelle_spectre.png', dpi=300, bbox_inches='tight')
plt.show()

# Exploration supplémentaire avec plusieurs facteurs d'échelle
scale_factors = [1.0, 0.8, 0.6, 0.4, 0.2, 0.1]
plt.figure(figsize=(15, 12))

# Création d'une grille pour afficher les différents facteurs d'échelle
for i, factor in enumerate(scale_factors):
    # Redimensionner l'image
    new_h, new_w = int(height * factor), int(width * factor)
    temp_img = cv2.resize(image_I, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled_img = cv2.resize(temp_img, (width, height), interpolation=cv2.INTER_LINEAR)
    
    # Calcul du spectre d'amplitude
    fft_scaled = np.fft.fft2(scaled_img)
    spectrum_scaled = np.abs(fft_scaled)
    
    # Affichage de l'image mise à l'échelle
    plt.subplot(3, 4, 2*i + 1)
    plt.imshow(scaled_img, cmap='gray')
    plt.title(f'Échelle: {factor:.1f}')
    plt.axis('off')
    
    # Affichage du spectre d'amplitude correspondant
    plt.subplot(3, 4, 2*i + 2)
    plt.imshow(display_spectrum(spectrum_scaled), cmap='viridis')
    plt.title(f'Spectre (échelle {factor:.1f})')
    plt.axis('off')

plt.tight_layout()
plt.savefig('comparaison_echelles_spectres.png', dpi=300, bbox_inches='tight')
plt.show()

# Approche alternative: agrandissement et réduction directe
plt.figure(figsize=(15, 12))

# Facteurs d'échelle (agrandissement et réduction)
scale_factors_zoom = [0.5, 0.75, 1.0, 1.5, 2.0]

for i, factor in enumerate(scale_factors_zoom):
    # Calculer les nouvelles dimensions
    new_h, new_w = int(height * factor), int(width * factor)
    
    # Redimensionner l'image
    if factor <= 1.0:
        # Réduction
        scaled_img = cv2.resize(image_I, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # Agrandissement
        scaled_img = cv2.resize(image_I, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Calculer la FFT (ajuster les dimensions pour la comparaison si nécessaire)
    fft_scaled = np.fft.fft2(scaled_img)
    spectrum_scaled = np.abs(fft_scaled)
    
    # Afficher l'image mise à l'échelle
    plt.subplot(3, 4, 2*i + 1)
    plt.imshow(scaled_img, cmap='gray')
    plt.title(f'Facteur: {factor:.2f}')
    plt.axis('off')
    
    # Afficher le spectre d'amplitude
    plt.subplot(3, 4, 2*i + 2)
    plt.imshow(display_spectrum(spectrum_scaled), cmap='viridis')
    plt.title(f'Spectre (facteur {factor:.2f})')
    plt.axis('off')

plt.tight_layout()
plt.savefig('zoom_et_spectres.png', dpi=300, bbox_inches='tight')
plt.show()