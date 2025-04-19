import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import math

def log_polar_transform_cv2(image, center=None, flags=cv2.WARP_FILL_OUTLIERS):
    """
    Applique la transformation log-polaire en utilisant OpenCV.
    
    Arguments:
        image: Image d'entrée (niveaux de gris ou RGB)
        center: Centre de la transformation (par défaut: centre de l'image)
        flags: Indicateurs pour cv2.logPolar (par défaut: cv2.WARP_FILL_OUTLIERS)
        
    Retourne:
        L'image transformée en coordonnées log-polaires
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    
    if center is None:
        center = (width // 2, height // 2)
    
    # Calculer la distance maximale du centre au coin le plus éloigné
    max_distance = np.sqrt((width - center[0])**2 + (height - center[1])**2)
    
    # Appliquer la transformation log-polaire
    log_polar_img = cv2.logPolar(image, center, max_distance, flags)
    
    return log_polar_img

def rotate_image(image, angle):
    """
    Fait pivoter une image autour de son centre.
    
    Arguments:
        image: Image d'entrée
        angle: Angle de rotation en degrés (sens antihoraire)
        
    Retourne:
        L'image pivotée
    """
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape
    
    center = (width // 2, height // 2)
    
    # Créer la matrice de rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Appliquer la rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
    
    return rotated_image

def demo_log_polar_rotation(input_image_path, rotation_angles=[0, 30, 60, 90]):
    """
    Démontre l'effet de la rotation sur la transformation log-polaire.
    
    Arguments:
        input_image_path: Chemin de l'image d'entrée
        rotation_angles: Liste d'angles de rotation à démontrer (en degrés)
    """
    # Charger l'image
    image = cv2.imread(input_image_path)
    
    # Convertir en niveaux de gris si l'image est en couleur
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()
    
    # Nombre d'angles de rotation
    n_angles = len(rotation_angles)
    
    # Créer une figure pour afficher les résultats
    fig, axes = plt.subplots(2, n_angles, figsize=(15, 8))
    
    for i, angle in enumerate(rotation_angles):
        # Faire pivoter l'image
        rotated_image = rotate_image(gray_image, angle)
        
        # Appliquer la transformation log-polaire
        log_polar_image = log_polar_transform_cv2(rotated_image)
        
        # Afficher l'image pivotée
        axes[0, i].imshow(rotated_image, cmap='gray')
        axes[0, i].set_title(f'Rotation {angle}°')
        axes[0, i].axis('off')
        
        # Afficher la transformation log-polaire
        axes[1, i].imshow(log_polar_image, cmap='gray')
        axes[1, i].set_title(f'Log-Polaire (Rotation {angle}°)')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('log_polar_rotation.png', dpi=300)
    plt.show()

# Exemple d'utilisation
if __name__ == "__main__":
    # Remplacer par le chemin de votre image
    input_image_path = "images/img_org_translated_img.PNG"
    
    # Démontrer la transformation log-polaire avec différentes rotations
    demo_log_polar_rotation(input_image_path, rotation_angles=[0, 45, 90, 180])