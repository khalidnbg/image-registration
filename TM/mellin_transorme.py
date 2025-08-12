import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.transform import rescale
from skimage.util import img_as_float

# Paramètres
scale_factor = 2.0
radius = 1500

# Chargement et conversion de l'image en niveaux de gris
image = img_as_float(data.astronaut().mean(axis=-1))
image_scaled = rescale(image, scale_factor, anti_aliasing=True)

# Fonction pour calculer la transformée de Mellin 1D le long des rayons
def mellin_transform(image, radius, beta=0.5):
    h, w = image.shape
    center = (h // 2, w // 2)
    r = np.logspace(0, np.log10(radius), 100)  # Échelle logarithmique
    mellin = []
    
    for r_val in r:
        integral = 0
        for theta in np.linspace(0, 2 * np.pi, 360):
            x = center[1] + r_val * np.cos(theta)
            y = center[0] + r_val * np.sin(theta)
            if 0 <= x < w and 0 <= y < h:
                integral += image[int(y), int(x)] * (r_val ** (-beta))
        mellin.append(integral)
    return np.array(mellin)

# Calcul de la transformée de Mellin pour les deux images
mellin_original = mellin_transform(image, radius)
mellin_scaled = mellin_transform(image_scaled, radius)

# Corrélation pour estimer le décalage d'échelle
from scipy.signal import correlate
cross_corr = correlate(mellin_original, mellin_scaled, mode='same')
shift = np.argmax(cross_corr) - len(cross_corr) // 2
klog = radius / np.log(radius)
recovered_scale = np.exp(-shift / klog)

# Affichage des résultats
print(f"Facteur d'échelle attendu : {scale_factor}")
print(f"Facteur d'échelle récupéré : {recovered_scale:.3f}")

# Visualisation
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(mellin_original, label='Original')
plt.plot(mellin_scaled, label='Redimensionnée')
plt.title("Transformée de Mellin")
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(cross_corr)
plt.title("Corrélation croisée")
plt.tight_layout()
plt.show()