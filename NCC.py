# Importation des bibliothèques nécessaires
import numpy as np
import cv2
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def ncc_cost(params, img_ref, img_mov, mask=None):
    """Fonction qui calcule le coût NCC (on veut maximiser NCC, donc minimiser -NCC)"""
    # Récupération des paramètres de transformation
    tx, ty, angle = params  # translation en x, y et rotation
    
    # Création de la matrice de transformation (rotation autour du centre)
    M = cv2.getRotationMatrix2D((img_mov.shape[1]//2, img_mov.shape[0]//2), angle, 1.0)
    # Ajout de la translation à la matrice
    M[0, 2] += tx  # translation en x
    M[1, 2] += ty  # translation en y
    
    # Application de la transformation à l'image mobile
    img_transformed = cv2.warpAffine(img_mov, M, (img_ref.shape[1], img_ref.shape[0]))
    
    # Application du masque si fourni
    if mask is not None:
        img1_masked = img_ref * mask
        img2_masked = img_transformed * mask
    else:
        img1_masked = img_ref
        img2_masked = img_transformed
    
    # Calcul des moyennes des images
    mean1 = np.mean(img1_masked)  # moyenne de l'image de référence
    mean2 = np.mean(img2_masked)  # moyenne de l'image transformée
    
    # Soustraction des moyennes (centrage des données)
    img1_centered = img1_masked - mean1
    img2_centered = img2_masked - mean2
    
    # Calcul du numérateur : somme des produits croisés
    numerator = np.sum(img1_centered * img2_centered)
    
    # Calcul du dénominateur : racine du produit des variances
    sum_sq1 = np.sum(img1_centered**2)  # somme des carrés image 1
    sum_sq2 = np.sum(img2_centered**2)  # somme des carrés image 2
    denominator = np.sqrt(sum_sq1 * sum_sq2)
    
    # Éviter la division par zéro
    if denominator == 0:
        return 1.0  # mauvais score si dénominateur nul
    
    # Calcul de la corrélation croisée normalisée
    ncc = numerator / denominator
    
    # Retourner -NCC car on veut maximiser NCC (minimize cherche le minimum)
    return -ncc

def register_images_ncc(img_ref, img_mov, initial_params=[0, 0, 0]):
    """Fonction principale qui trouve les meilleurs paramètres de recalage par NCC"""
    
    # Conversion des images couleur en niveaux de gris
    if len(img_ref.shape) == 3:  # si l'image a 3 canaux (RGB)
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    # Normalisation des valeurs entre 0 et 1 (au lieu de 0-255)
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    
    # Optimisation : cherche les paramètres qui maximisent NCC (minimisent -NCC)
    result = minimize(ncc_cost, initial_params,  # commence avec [0,0,0]
                     args=(img_ref, img_mov),    # passe les images à la fonction
                     method='Powell',            # algorithme d'optimisation
                     options={
                        'maxiter': 1000,  # Augmenter le nombre d'itérations
                        'ftol': 1e-4,    # Tolérance plus stricte
                        'disp': True
                    })
    
    return result.x  # retourne les meilleurs paramètres trouvés

def apply_transformation(img, params):
    """Applique la transformation trouvée à une image"""
    tx, ty, angle = params  # récupère les paramètres
    
    # Création de la même matrice de transformation
    M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Application de la transformation
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

def calculate_ncc_score(img_ref, img_mov):
    """Calcule le score NCC final entre deux images (pour vérification)"""
    # Conversion en niveaux de gris si nécessaire
    if len(img_ref.shape) == 3:
        img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    if len(img_mov.shape) == 3:
        img_mov = cv2.cvtColor(img_mov, cv2.COLOR_BGR2GRAY)
    
    # Normalisation
    img_ref = img_ref.astype(np.float32) / 255.0
    img_mov = img_mov.astype(np.float32) / 255.0
    
    # Calcul NCC
    return -ncc_cost([0, 0, 0], img_ref, img_mov)

# Script principal
if __name__ == "__main__":
    img_ref = cv2.imread('images/tisdrin.png')  # image fixe
    img_mov = cv2.imread('images/tisdrin_translated_12_27.jpg')     # image à recaler
    
    # Score NCC avant recalage
    ncc_before = calculate_ncc_score(img_ref, img_mov)
    print(f"Score NCC avant recalage: {ncc_before:.4f}")
    
    # 2. Lancer le recalage (trouve les meilleurs paramètres)
    params = register_images_ncc(img_ref, img_mov)
    print(f"Paramètres trouvés - tx: {params[0]:.2f}, ty: {params[1]:.2f}, angle: {params[2]:.2f}°")
    
    # 3. Appliquer la transformation à l'image mobile
    img_registered = apply_transformation(img_mov, params)
    
    # Score NCC après recalage
    ncc_after = calculate_ncc_score(img_ref, img_registered)
    print(f"Score NCC après recalage: {ncc_after:.4f}")
    print(f"Amélioration: {ncc_after - ncc_before:.4f}")
    
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
    plt.title(f'Image mobile (avant)\nNCC: {ncc_before:.3f}')
    plt.axis('off')
    
    # Image mobile après recalage
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(img_registered, cv2.COLOR_BGR2RGB))
    plt.title(f'Image mobile (après recalage)\nNCC: {ncc_after:.3f}')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()