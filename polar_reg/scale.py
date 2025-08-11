import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rescale
from skimage.util import img_as_float

# Paramètres modifiables
radius = 1500  # Doit être assez grand pour capturer l'image entière
scale = 2.2    # Facteur d'échelle à appliquer (vous pouvez le changer)

# Chargement de l'image (remplacez par votre propre image si nécessaire)
image = data.astronaut()  # Vous pouvez utiliser data.camera() pour une image N&B
image = img_as_float(image)

# Application du changement d'échelle seulement
rescaled = rescale(image, scale, channel_axis=-1)

# Transformation en coordonnées polaires logarithmiques
image_polar = warp_polar(image, radius=radius, scaling='log', channel_axis=-1)
rescaled_polar = warp_polar(rescaled, radius=radius, scaling='log', channel_axis=-1)

# Affichage
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()
ax[0].set_title("Original")
ax[0].imshow(image)
ax[1].set_title("Redimensionné (x{})".format(scale))
ax[1].imshow(rescaled)
ax[2].set_title("Log-Polar-Transformed Original")
ax[2].imshow(image_polar)
ax[3].set_title("Log-Polar-Transformed Rescaled")
ax[3].imshow(rescaled_polar)
plt.tight_layout()
plt.show()

# Calcul de la corrélation de phase avec upsampling pour plus de précision
shifts, error, phasediff = phase_cross_correlation(
    image_polar, rescaled_polar, upsample_factor=20, normalization=None
)

# Extraction du décalage vertical (qui correspond au changement d'échelle)
shiftc = shifts[1]

# Calcul du facteur d'échelle à partir du décalage
klog = radius / np.log(radius)
recovered_scale = 1 / (np.exp(shiftc / klog))

print(f'Expected scaling factor: {scale}')
print(f'Recovered scaling factor: {recovered_scale}')