import numpy as np
import matplotlib.pyplot as plt
from skimage import io, data
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
from skimage.util import img_as_float

def detect_rotation_and_scale(original_image, angle=35, scale=1.0, radius=None, use_log_polar=True):
    """
    Détecte la rotation et le changement d'échelle entre une image originale et sa version transformée.
    
    Args:
        original_image: image d'entrée (numpy array)
        angle: angle de rotation à appliquer (degrés)
        scale: facteur d'échelle à appliquer
        radius: rayon pour la transformation polaire (si None, calculé automatiquement)
        use_log_polar: utiliser la transformation log-polaire pour détecter l'échelle
    
    Returns:
        Tuple contenant (angle détecté, échelle détectée)
    """
    # Convertir l'image en float
    image = img_as_float(original_image)
    
    # Déterminer le rayon si non spécifié
    if radius is None:
        radius = max(image.shape) // 2
    
    # Appliquer les transformations
    rotated = rotate(image, angle)
    transformed = rescale(rotated, scale, channel_axis=-1 if image.ndim == 3 else None)
    
    # Transformation polaire
    if use_log_polar:
        image_polar = warp_polar(image, radius=radius, scaling='log', 
                                channel_axis=-1 if image.ndim == 3 else None)
        transformed_polar = warp_polar(transformed, radius=radius, scaling='log', 
                                     channel_axis=-1 if image.ndim == 3 else None)
    else:
        image_polar = warp_polar(image, radius=radius, 
                               channel_axis=-1 if image.ndim == 3 else None)
        transformed_polar = warp_polar(transformed, radius=radius, 
                                    channel_axis=-1 if image.ndim == 3 else None)
    
    # Affichage
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax = axes.ravel()
    ax[0].set_title("Original")
    ax[0].imshow(image)
    ax[1].set_title(f"Transformée (rotation={angle}°, echelle={scale}x)")
    ax[1].imshow(transformed)
    ax[2].set_title("Polar-Transformed Original")
    ax[2].imshow(image_polar)
    ax[3].set_title("Polar-Transformed Transformed")
    ax[3].imshow(transformed_polar)
    plt.show()
    
    # Calcul des décalages
    shifts, error, phasediff = phase_cross_correlation(
        image_polar, transformed_polar, upsample_factor=20, normalization=None
    )
    
    # Extraction des résultats
    detected_angle = shifts[0]
    
    if use_log_polar:
        klog = radius / np.log(radius)
        detected_scale = 1 / (np.exp(shifts[1] / klog))
    else:
        detected_scale = 1.0  # Sans log-polar, on ne peut pas détecter l'échelle
    
    return detected_angle, detected_scale

# Paramètres
angle = 53.7  # Angle de rotation en degrés
scale = 2.2   # Facteur d'échelle
radius = 1500 # Rayon pour la transformation polaire

# Charger une image (remplacez par votre propre image si nécessaire)
# image = io.imread("votre_image.jpg")
image = data.retina()

# Détection de la rotation et de l'échelle
detected_angle, detected_scale = detect_rotation_and_scale(
    image, angle=angle, scale=scale, radius=radius, use_log_polar=True
)

# Affichage des résultats
print("\nRésultats de la détection:")
print(f"Angle attendu: {angle}°")
print(f"Angle détecté: {detected_angle}°")
print(f"Échelle attendue: {scale}x")
print(f"Échelle détectée: {detected_scale:.4f}x")
print(f"Erreur angle: {abs(angle - detected_angle):.2f}°")
print(f"Erreur échelle: {abs(scale - detected_scale):.4f}x")