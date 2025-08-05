import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def ssd_cost(params, img_ref, img_mov, mask=None):
    """Fonction qui calcule le coût SSD (plus c'est petit, mieux c'est)"""
    # Récupération des paramètres de transformation
    tx, ty, angle = params
    
    # Création de la matrice de transformation (rotation autour du centre)
    M = cv2.getRotationMatrix2D((img_mov.shape[1]//2, img_mov.shape[0]//2), angle, 1.0)
    # Ajout de la translation à la matrice
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Application de la transformation à l'image mobile
    img_transformed = cv2.warpAffine(img_mov, M, (img_ref.shape[1], img_ref.shape[0]))
    
    # Calcul de la différence entre les images
    if mask is not None:
        diff = (img_ref - img_transformed) * mask
    else:
        diff = img_ref - img_transformed
    
        print(np.sum(diff**2))
        
        return np.sum(diff**2)

def register_images_ssd(img_ref, img_mov, initial_params=[0, 0, 0]):
    """Fonction principale qui trouve les meilleurs paramètres de recalage"""
    
    # Conversion des images couleur en niveaux de gris
    if len(img_ref.shape) == 3: # si l'image a 3 canaux (RGB)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    # Normalisation des valeurs entre 0 et 1 (au lieu de 0-255)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    
    # Optimisation : cherche les paramètres qui minimisent le coût SSD
    result = minimize(ssd_cost, initial_params, # commence avec [0,0,0]
                     args=(img_ref, img_mov), # passe les images à la fonction
                     method='Powell', # algorithme d'optimisation
                     options={'maxiter': 1000}) # maximum 1000 itérations
    
    return result.x  # retourne les meilleurs paramètres trouvés

def apply_transformation(img, params):
    """Applique la transformation trouvée à une image"""
    tx, ty, angle = params  # récupère les paramètres
    
    # Création de la même matrice de transformation
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

# Exemple d'utilisation
if __name__ == "__main__":
    # Charger vos images
    img_ref = cv2.imread('images/number2.png')
    img_mov = cv2.imread('images/number2_rotated.png')
    
    # 2. Lancer le recalage (trouve les meilleurs paramètres)
    params = register_images_ssd(img_ref, img_mov)
    print(f"Paramètres trouvés - tx: {params[0]:.2f}, ty: {params[1]:.2f}, angle: {params[2]:.2f}°")
    
    # 3. Appliquer la transformation à l'image mobile
    img_registered = apply_transformation(img_mov, params)
    
    # 4. Affichage des résultats (avant/après)
    plt.figure(figsize=(15, 5))
    
    # Image de référence
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB))
    plt.title('Image de référence')
    plt.axis('off')
    
    # Image mobile originale
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(img_mov, cv2.COLOR_BGR2RGB))
    plt.title('Image mobile (avant)')
    plt.axis('off')
    
    # Image mobile après recalage
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_registered, cv2.COLOR_BGR2RGB))
    plt.title('Image mobile (après recalage)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()